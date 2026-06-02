import numpy as np
from olmo import tokenizer
import re
from olmo.preprocessing.text_preprocessor import InterleavedTextPreprocessor
from olmo.tokenizer import build_tokenizer, IMAGE_PROMPT


def _remove_video_text(text):
    return re.sub(fr"(time [0-9]|FPS)(.*{tokenizer.IM_END_TOKEN})+", tokenizer.IMAGE_PROMPT, text)


def get_preprocessor():
    tokenizer = build_tokenizer("Qwen/Qwen2-7B")
    return InterleavedTextPreprocessor(tokenizer=tokenizer)


def _test_tokenization(messages, n_at_start=None, preprocessor=None):
    if preprocessor is None:
        preprocessor = get_preprocessor()
    is_multi_message = isinstance(messages[0], list)
    if n_at_start is None and is_multi_message:
        n = " ".join(messages[0]).count(IMAGE_PROMPT)
    elif n_at_start is None:
        n = " ".join(messages).count(IMAGE_PROMPT)
    else:
        n = n_at_start
    tok = preprocessor.tokenizer
    batch = preprocessor.tokenize_and_interleave(messages, [[tok.image_prompt_token_id]]*n)

    if not is_multi_message:
        with_tokens = tok.decode(batch["input_tokens"], truncate_at_eos=False)
        if n_at_start is None:
            expected = "".join(messages)
        else:
            expected = "".join([tokenizer.IMAGE_PROMPT]*n + messages)
        assert batch["input_tokens"][0] == tok.bos_token_id
        assert batch["target_tokens"][-1] == tok.eos_token_id
        assert batch["loss_masks"][-1] == 1.0
        assert with_tokens == expected, f"Expected \"{expected}\", but got \"{with_tokens}\""
        assert np.all(batch["position_ids"] == np.arange(len(batch["input_tokens"])))
        assert tok.decode(batch["target_tokens"][batch["loss_masks"] > 0], False) == "".join(messages[1::2])
    else:
        subsegments = batch["subsegment_ids"]
        pos_ids = batch["position_ids"]
        for ix, seg_messages in enumerate(messages):
            expected = "".join(seg_messages)
            if tokenizer.IMAGE_PROMPT not in expected:
                expected = tokenizer.IMAGE_PROMPT + expected
            msg_mask = (subsegments == ix) | (subsegments == 10000)
            seg_tokens = batch["input_tokens"][msg_mask]
            seg_targets = batch["target_tokens"][msg_mask]
            seg_loss = batch["loss_masks"][msg_mask]

            assert (seg_tokens == tok.eos_token_id).sum() == len(seg_messages)//2
            assert np.all(pos_ids[msg_mask] - pos_ids[msg_mask][0] == np.arange(msg_mask.sum()))
            assert seg_tokens[0] == tok.bos_token_id
            assert seg_targets[-1] == tok.eos_token_id
            assert seg_loss[-1] == 1.0
            if n_at_start is not None:
                actual = tok.decode(seg_tokens, False)
                assert _remove_video_text(actual) == expected
            assert tok.decode(seg_targets[seg_loss > 0], False) == "".join(seg_messages[1::2])


def test_text_only():
    _test_tokenization(["What time is it?", " 3"])
    _test_tokenization(["a b", " res1 res2", " d e", " res5"])


def test_max_tokens():
    preprocessor = get_preprocessor()
    tok = preprocessor.tokenizer
    preprocessor.max_text_tokens = 2
    batch = preprocessor.tokenize_and_interleave(
        [
            ["question1", " answer1"],
            ["question2", " answer2"],
            ["question1", " answer2"*100]
        ],
        [[preprocessor.tokenizer.image_prompt_token_id]]
    )
    assert tok.decode(batch["input_tokens"], False) == f"{tokenizer.IMAGE_PROMPT}question1 answer1"

    preprocessor.max_text_tokens = 8
    batch = preprocessor.tokenize_and_interleave(
        [
            ["question1", " answer1"],
            ["question2", " answer2"],
            ["question1", " answer2"*100]
        ],
        [[preprocessor.tokenizer.image_prompt_token_id]]
    )
    assert tok.decode(batch["input_tokens"], False) == f"{tokenizer.IMAGE_PROMPT}question1 answer1question2 answer2"


def test_max_seq_len():
    rng = np.random.RandomState(123)
    preprocessor = get_preprocessor()
    messages = [
        [" a"*rng.randint(1, 5), " b"*rng.randint(1, 5)]
        for i in range(20)
    ]
    n_mm_tokens = 27
    mm_tokens = [np.zeros([n_mm_tokens], dtype=np.int32)]
    max_len = len(preprocessor.tokenizer.encode("".join("".join(x) for x in messages))) + n_mm_tokens
    for max_seq_len in range(n_mm_tokens+2, max_len+2):
        preprocessor.max_sequence_length = max_seq_len
        batch = preprocessor.tokenize_and_interleave(messages, mm_tokens)
        last_ids = batch["subsegment_ids"][max_seq_len:]
        if len(last_ids) != 0:
            # Overflow can happen, but should only be one segment, and should not truncate
            # all the loss tokens for that segment
            assert np.all(last_ids[0] == last_ids)
            assert batch["loss_masks"][batch["subsegment_ids"] == last_ids[0]][:max_seq_len].sum() > 0
        if max_seq_len >= max_len:
            # Should include all the messages
            assert np.any(batch["subsegment_ids"] == (len(messages)-1))


def test_at_start():
    _test_tokenization([" question?", " answer"], 1)
    _test_tokenization([" a long question", " ans"], 2)
    _test_tokenization([" a long question", " ans", " next", " end"], 2)


def test_in_message():
    _test_tokenization([f"look at {IMAGE_PROMPT} what is it?", "answer"])
    _test_tokenization([f"question", "answer", f"q2 {IMAGE_PROMPT}.", "answer2"])
    _test_tokenization(
        [f"1: {IMAGE_PROMPT} 2: {IMAGE_PROMPT} 3: {IMAGE_PROMPT}?", "answer"],
    )
    _test_tokenization([
        f"1: {IMAGE_PROMPT} 2: {IMAGE_PROMPT} 3: {IMAGE_PROMPT}?",
        "response 1",
        f"4: {IMAGE_PROMPT}",
        "response 2",
    ])


def test_multi_message():
    _test_tokenization([
        [" turn11", " turn12", " turn13", " turn14"],
        [" turn21", " turn22"],
        [" turn31", " turn32", " turn33", " turn34"],
    ], 1)


def test_multi_message_in_middle():
    _test_tokenization([
        [f" turn11 {IMAGE_PROMPT} debeug", " turn12", f" turn13.", " turn14"],
        [f"{IMAGE_PROMPT} turn21", " turn22"],
        [f" turn31 some {IMAGE_PROMPT} text", " turn32", " turn33", " turn34"],
    ])

