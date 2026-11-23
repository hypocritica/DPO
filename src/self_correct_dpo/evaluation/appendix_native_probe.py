"""SD1.5 observations for native DMPO/DSPO, sharing the original input pipeline."""
import numpy as np
from self_correct_dpo.evaluation.appendix_probe import PairProbe, seed_for
from self_correct_dpo.evaluation.appendix_scan import scan_seed
from self_correct_dpo.evaluation.appendix_native_gradients import coefficients, collect_native_statistics, loss_value, objective_spec


class NativePairProbe(PairProbe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec = objective_spec(self.backend.training_config)

    def moment_tensors(self, condition, seeds):
        import torch
        from self_correct_dpo.algorithms.formal import _repeat_condition
        start, stop = condition["index_start"], condition["index_stop"]
        latent = self.latents[0, 0]
        noises, indices = [], []
        for seed in seeds:
            generator = torch.Generator(device=latent.device).manual_seed(seed)
            noises.append(torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=generator))
            indices.append(torch.full((1,), start, device=latent.device, dtype=torch.long) if stop-start == 1
                           else torch.randint(start, stop, (1,), device=latent.device, generator=generator))
        indices = torch.cat(indices)
        latents = self.latents.expand(len(seeds), *self.latents.shape[1:])
        noisy, times, target = self.backend.pair_view_inputs(latents, torch.stack(noises)[:,None].expand_as(latents),
                                                           indices[:,None].expand(-1,2))
        text = self.condition if len(seeds) == 1 else _repeat_condition(self.condition, len(seeds))
        was_training = self.backend.model.training
        self.backend.model.eval()
        try:
            with torch.no_grad(), self.backend.model.disable_adapter(), self.backend._autocast():
                reference = self.backend._predict(noisy, times, text)
        finally:
            self.backend.model.train(was_training)
        with self.backend._autocast():
            prediction = self.backend._predict(noisy, times, text)
        policy_errors = (prediction.float()-target.float()).square().flatten(1).mean(1).reshape(-1,2)
        reference_errors = (reference.float()-target.float()).square().flatten(1).mean(1).reshape(-1,2).detach()
        if self.spec["name"] == "dmpo":
            relative = reference_errors-policy_errors
            u = self.beta * (relative[:,0]-relative[:,1])
            return u[:,None], indices
        # Match archived score_residual: reduce moments in FP32, then form gate in FP64.
        gap = (prediction[::2].float()-reference[::2].detach().float()).square().flatten(1).mean(1)
        relative = reference_errors.double()-policy_errors.double()
        m = .5*self.beta*(relative[:,0]-relative[:,1])
        return torch.stack((m, policy_errors[:,0].double(), gap.double(), reference_errors[:,0].double()),dim=1), indices

    def stat_gradients(self, condition, seeds):
        import torch
        values, indices = self.moment_tensors(condition, seeds)
        components = len(self.spec["components"])
        for i in range(len(seeds)):
            rows = []
            for j in range(components):
                last = i == len(seeds)-1 and j == components-1
                grads = torch.autograd.grad(values[i,j], self.parameters, retain_graph=not last)
                rows.append(torch.cat([g.detach().reshape(-1).float() for g in grads]))
            yield values[i].detach().double().cpu().numpy(), torch.stack(rows), int(indices[i])

    def gradients(self, condition, seeds):
        import torch
        for x,h,index in self.stat_gradients(condition,seeds):
            c = torch.as_tensor(coefficients(x,self.spec),dtype=h.dtype,device=h.device)
            yield float(loss_value(x,self.spec)), (c[:,None]*h).sum(0), index

    def gradient(self, condition, seed):
        return next(self.gradients(condition,[seed]))


def initialization_native(backend, beta, rows, pairs, conditions, checkpoint_id, view_seed, init_seed):
    import torch
    parameters = tuple(p for _,p in sorted((n,p) for n,p in backend.model.named_parameters() if p.requires_grad))
    pair = pairs[0]
    probe = NativePairProbe(backend,parameters,beta,rows[0],pair['prompt'],pair['latent_seed'])
    checks = []
    for condition in conditions:
        seeds = [seed_for('initialization-native',checkpoint_id,view_seed,pair['pair_id'],condition['condition'],i) for i in range(2)]
        views = list(probe.stat_gradients(condition,seeds))
        x = np.stack([v[0] for v in views]); h = torch.stack([v[1] for v in views])
        c = torch.as_tensor(coefficients(x,probe.spec),dtype=h.dtype,device=h.device)
        b = (h*c[:,:,None]).sum(1).mean(0)
        cm = torch.as_tensor(coefficients(x.mean(0),probe.spec),dtype=h.dtype,device=h.device)
        aggregate = (h.mean(0)*cm[:,None]).sum(0)
        delta = float((b-aggregate).norm()); norm=max(float(b.norm()),float(aggregate.norm()))
        agrees_with_base = float(np.max(np.abs(x[:,0]))) <= 1e-5
        if probe.spec['name']=='dspo':
            agrees_with_base = agrees_with_base and float(np.max(np.abs(x[:,2]))) <= 1e-7 and np.allclose(x[:,1],x[:,3],rtol=0,atol=1e-7)
        passed = bool(agrees_with_base and norm>0 and delta <= 1e-7+1e-5*norm)
        checks.append(dict(condition=condition['condition'],max_abs_primary_moment=float(np.max(np.abs(x[:,0]))),
                           lossavg_gradient_norm=float(b.norm()),mvpo_gradient_norm=float(aggregate.norm()),
                           gradient_difference_norm=delta,passed=passed))
    return dict(native_objective=probe.spec,initial_lora_seed=init_seed,checks=checks,passed=all(c['passed'] for c in checks))


def native_measurement(probe, condition, sampling, pair_id, scratch_dir, scan):
    source, audit = [
        [[scan_seed(scan['rng_scope'],'view',sampling['view_seed'],pair_id,condition['condition'],role,b,i)
          for i in range(sampling[f'{role}_views_per_block'])] for b in range(sampling[f'{role}_blocks'])]
        for role in ('source','audit')]
    size = scan['view_batch_size']
    observed = {}
    def observations():
        for block in source+audit:
            for start in range(0,len(block),size):
                seeds=block[start:start+size]
                for seed,(x,h,index) in zip(seeds,probe.stat_gradients(condition,seeds)):
                    yield seed,x,h,index
    stream=observations()
    def observe(seed):
        actual,x,h,index=next(stream)
        if actual!=seed:
            raise ValueError('native collector order differs from frozen view bank')
        observed[seed]=dict(seed=seed,scheduler_index=index,margin=float(x[0]),moments=x.tolist())
        return x,h
    raw=collect_native_statistics(observe,source_seeds=source,audit_seeds=audit,ks=sampling['ks'],spec=probe.spec,scratch_dir=scratch_dir)
    return dict(statistics=raw,source_views=[[observed[s] for s in b] for b in source],
                audit_views=[[observed[s] for s in b] for b in audit])
