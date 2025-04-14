import json
from typing import Dict, Any
import os
from config import load_config
from core.llm.llm_chat import AzureLLMChat
from qa_app import QAApp
import pandas as pd
from tqdm import tqdm

from query.query_helper import QueryHelper

tqdm.pandas()


class Evaluator:
    """
        Class responsible for evaluating a QA system using ground truth data and LLM-based evaluation.
    """
    def __init__(self,
                 config: Dict[str, Any]):
        """Initialize the Evaluator with configuration settings.

        Args:
            config: Dictionary containing configuration parameters for evaluation
        """
        env_config = config.get("env_config", {})
        llm_config = config.get("llm_config", {})

        self.query_llm = AzureLLMChat(**env_config, **llm_config)
        self.query_helper = QueryHelper(self.query_llm)

        self.qa_app = QAApp(config)
        self.eval_json_path = config["eval_config"]["data_path"]
        self.eval_out_path = config["eval_config"]["output_path"]

        if not os.path.exists(self.eval_out_path):
            self._load_eval_dataset()
        else:
            self.eval_df = pd.read_csv(self.eval_out_path)

    def _load_eval_dataset(self):
        """Load and process evaluation dataset from JSON file."""

        with open(self.eval_json_path, 'r') as f:
            data = json.load(f)

        records = []
        for topic, qa_pair_list in data.items():
            for qa_pair in qa_pair_list:
                records.append(
                    {
                        "Topic": topic,
                        "Question": qa_pair["question"],
                        "GroundTruth": qa_pair["answer"]
                    }
                )
        self.eval_df = pd.DataFrame(records)

    def evaluate(self):
        """Run the complete evaluation pipeline."""

        self.answer_questions()
        self.llm_as_a_judge_acc()
        self.llm_as_a_judge_relevance()

    def answer_questions(self):
        """Process each question through the QA system and store the answers."""

        self.eval_df["QARagAnswer"] = \
            self.eval_df.progress_apply(lambda row: self.qa_app.process_query(row["Question"],
                                                                              None,
                                                                              eval=True)[0],
                                        axis=1)

        self.eval_df.to_csv(self.eval_out_path)

    def llm_as_a_judge_acc(self):
        """Evaluate accuracy of answers compared to ground truth."""

        self.eval_df["AccuracyScore"] = self.eval_df.progress_apply(lambda row:
                                                                    self.query_helper.llm_as_a_judge_accuracy_query(
                                                                        row["GroundTruth"],
                                                                        row["QARagAnswer"]
                                                                    ), axis=1)
        self.eval_df.to_csv(self.eval_out_path)

    def llm_as_a_judge_relevance(self):
        """Evaluate relevance of answers to the questions."""

        self.eval_df["RelevanceScore"] = self.eval_df.progress_apply(lambda row:
                                                                     self.query_helper.llm_as_a_judge_relevance_query(
                                                                         row["Question"],
                                                                         row["QARagAnswer"]
                                                                     ), axis=1)
        self.eval_df.to_csv(self.eval_out_path)


if __name__ == "__main__":
    config = load_config()
    eval = Evaluator(config)
    eval.evaluate()