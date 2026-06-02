import datasets

from olmo.data.dataset import Dataset


class Tulu4Filtered(Dataset):
    """
    NLP data used in Tulu4, classified into code/non-english categories so that
    it can be filtered for Molmo training
    """

    @classmethod
    def format_messages(cls, parts):
        messages = []
        if parts[0]["role"] == "system":
            assert parts[1]["role"] == "user"
            assert "\n" not in parts[0]['content']
            messages.append(f"System: {parts[0]['content']}\n{parts[1]['content']}")
            parts = parts[2:]
        elif parts[0]["role"] == "assistant":
            return None
        else:
            messages.append(parts[0]['content'])
            parts = parts[1:]

        for ix, message in enumerate(parts):
            if ix % 2 == 0:
                if message["role"] != "assistant":
                    return None
            else:
                if message["role"] != "user":
                    return None
            messages.append(message["content"])
        if len(messages) <= 1:
            return None
        return messages

    def __init__(self, split: str, use_code=False, use_puzzles=False, use_reasoning=False, use_non_english=False, max_first_msg_len=4096):
        self.data = datasets.load_dataset(
            "allenai/molmo2-tulu4-classified",
            split=split,
            keep_in_memory=False
        )

        def _filter(cls, src, n_tokens, empty_message, has_special_token):
            if empty_message or has_special_token:
                return False
            if src in ["allenai/dino-hardcodes", "allenai/hardcoded-olmo"]:
                return False
            if not use_puzzles and src == "allenai/puzzle_data_160k-ngram-filtered":
                return False
            if not use_reasoning and src in [
                "faezeb/verifiable-reasoning-v3-o4-mini-length-filtered-verified",
                "allenai/verifiable-reasoning-filtered-o4-mini-filtered",
            ]:
                return False
            if not use_code and cls == "code":
                return False
            if not use_non_english and cls == "non-english":
                return False
            if max_first_msg_len and n_tokens > max_first_msg_len:
                return False
            return True
        self.data = self.data.filter(_filter, input_columns=[
            "category", "source", "first_message_qwen3_tokens", "empty_messages", "has_special_token"])

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        messages = self.format_messages(ex["messages"])
        assert messages is not None
        return dict(
            message_list=[dict(messages=messages, style="text_sft")],
            metadata=dict(
                example_id=ex["id"],
                source=ex["source"],
            )
        )