import argparse
import logging

from src.task import Pipeline

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="eval", choices=["infer", "eval"])
    parser.add_argument("--task", default="aqa", help="select a infer task to run")
    parser.add_argument("--model", default=None, help="select a model to generate")
    
    # NOTE(TTTdas): Global parameters below are optional and can be set via different config files. 
    # NOTE(TTTdas): If set, global values will override those in the config files.
    parser.add_argument("--bsz", default=None, help="batch size when generation and evaluation, usually defined in dataset.yaml and eval_task.yaml")
    parser.add_argument("--save_dir", default="", help="directory to save generation and evaluation results")
    parser.add_argument("--eval_task", default=None)
    parser.add_argument("--save_pred_audio", default=None, help="if False, generation only save the text response")
    parser.add_argument("--keep_meta_info", default=True, help="if False, generation only save infomation from model output")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
        encoding="utf-8"
    )
    user_args = vars(args)
    logger.info(f"Processing task: \nglobal args: {user_args}")
    t = Pipeline.create(**user_args)
    t.run()

if __name__ == "__main__":
    main()
