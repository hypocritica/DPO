"""Train from a source checkout, without requiring an editable installation."""
import logging
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from self_correct_dpo.train import main


def _configure_sd3_prompt_logging():
    """Keep truncation counters without dumping entire captions every batch."""
    reported = False

    def keep(record):
        nonlocal reported
        prefixes = (
            "The following part of your input was truncated because CLIP can only handle sequences up to",
            "The following part of your input was truncated because `max_sequence_length` is set to",
        )
        if record.levelno != logging.WARNING or not record.getMessage().startswith(prefixes):
            return True
        if reported or os.environ.get("RANK", "0") != "0":
            return False
        reported = True
        record.msg = ("SD3 long prompts are truncated at the configured CLIP/T5 limits; "
                      "repeated prompt dumps are suppressed. "
                      "Counts: prompt_truncated_*_per_microbatch.")
        record.args = ()
        return True

    logging.getLogger("diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3").addFilter(keep)


if __name__ == "__main__":
    _configure_sd3_prompt_logging()
    main()
