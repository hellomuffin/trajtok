"""Class for doing prompting/other data formatting for tasks

For example, converting points to text, or applying prompt templates
"""
import dataclasses
import re
import random
import logging
from collections import Counter
import string
from typing import Optional, Dict, Tuple, List, Union

import numpy as np

from olmo import tokenizer
from olmo.config import BaseConfig
from olmo.preprocessing.multiple_choice_templates import template_mc_question
from olmo.preprocessing.point_formatter import LegacyPointFormatting, UnifiedPointFormatter, \
    PointFormatter
from olmo.util import parse_timestamp

GENERAL_PROMPTS_V1 = {
    "short_answer": [
        "Answer this question very briefly\n{question}",
        "{question} Answer with a few words",
        "{question} Respond very briefly",
        "{question} Answer directly without any details, explanation, or elaboration",
        "I have a question about this image, please answer it very briefly: {question}",
        "Question: {question} Short Answer:",
        "Question: {question}\nShort Answer:",
        '{question}\nAnswer the question as briefly as possible.',
        'Answer very briefly:\n{question}',
        'The question "{question}" can be answered using the image. A short answer is',
        "{question} Based on the image, respond to this question with a short answer:",
        "{question} Short answer:",
        "{question} A short answer to the question is",
        "Give a short, matter-of-fact answer to this question: {question}",
        "Give me a simple, direct answer to this question, do not elaborate or explain your answer:\n{question}"
    ],
    "short_caption": [
        'Caption the image with 1 or two sentences',
        'Write a very short description of this image.',
        'Briefly describe the image.',
        'Look and this image, and then summarize it in a sentence or two.',
        'Write a brief caption describing the image',
        'Brief Caption:',
        'A short image caption:',
        'A short image description',
        'Briefly describe the content of the image.',
        'Can you give me one sentence summary of the picture?',
        'How would you describe this image in a sentence or two?',
    ],
    "long_caption": [
        'Describe this image.',
        'Describe this image',
        'describe the image',
        'Write a long description of this image.',
        'caption the picture',
        'Caption',
        'caption',
        'Construct a long caption for this image',
        'Generate a caption',
        'Create a detailed caption',
        'Write a long caption',
        'Describe this image in detail',
        'Describe this',
        'describe this',
        'Caption this',
        'What can be seen in this image?',
        'What do you see in the image?',
        'Look at this photo carefully and then tell me about it in detail',
        'Write a long description of this image',
        'Tell me about this picture.',
        'Write a paragraph about this image.',
        'Look at this image carefully and then describe it in detail',
        'Generate a long caption about this image.'
    ],
    "video_motion_caption": [
        "Describe the motion in this video.",
        "Describe the motion in this clip",
        "describe the motion in the video",
        "Motion-focused caption",
        "motion-focused caption",
        "Create a detailed video caption centered on motion",
        "Describe this video in detail with attention to motion cues",
        "Caption this video with a focus on motion",
        "Tell me about the motion in this video.",
        "Watch the clip closely and describe the motion in detail",
    ],
    "video_object_caption": [
        "Caption the {object} in the video with 1 or two sentences.",
        "Write a very short description of the {object} in this video.",
        "Briefly describe the {object} in the video.",
        "Look at the {object} in this video, and then summarize it in a sentence or two.",
        "Write a brief caption describing the {object} in this video.",
        "Brief caption of the {object} in the video:",
        "A short video caption for the {object}:",
        "A short description of the {object} in the video.",
        "Can you give me one sentence summarizing the {object} in this video?",
        "How would you describe the {object} in this video in a sentence or two?"
],
    "video_clip_caption_start_end": [
        'Describe this video from {start_time} to {end_time}.',
        'describe the video from {start_time} to {end_time}.',
        'Write a long description of this video from {start_time} to {end_time}.',
        'caption the video from {start_time} to {end_time}.',
        'Caption {start_time} to {end_time}.',
        'caption {start_time} to {end_time}.',
        'Construct a long caption for this video from {start_time} to {end_time}.',
        'Generate a caption for {start_time} to {end_time}.',
        'Create a detailed caption for {start_time} to {end_time}.',
        'Write a long caption for {start_time} to {end_time}.',
        'Describe this video in detail from {start_time} to {end_time}.',
        'Describe this from {start_time} to {end_time}.',
        'describe this from {start_time} to {end_time}.',
        'Caption this from {start_time} to {end_time}.',
        'What can be seen in this video from {start_time} to {end_time}?',
        'What do you see in the video from {start_time} to {end_time}?',
        'Look at this video carefully from {start_time} to {end_time} and then tell me about it in detail',
        'Write a long description of this video from {start_time} to {end_time}.',
        'Tell me about this video from {start_time} to {end_time}.',
        'Write a paragraph about this video from {start_time} to {end_time}.',
        'Generate a long caption about this video from {start_time} to {end_time}.',
    ],
    "video_clip_caption_start_end_in_seconds": [
        'Describe this video from {start_time} to {end_time} seconds.',
        'describe the video from {start_time} to {end_time} seconds.',
        'Write a long description of this video from {start_time} to {end_time} seconds.',
        'caption the video from {start_time} to {end_time} seconds.',
        'Caption {start_time} to {end_time} seconds.',
        'caption {start_time} to {end_time} seconds.',
        'Construct a long caption for this video from {start_time} to {end_time} seconds.',
        'Generate a caption for {start_time} to {end_time} seconds.',
        'Create a detailed caption for {start_time} to {end_time} seconds.',
        'Write a long caption for {start_time} to {end_time} seconds.',
        'Describe this video in detail from {start_time} to {end_time} seconds.',
        'Describe this from {start_time} to {end_time} seconds.',
        'describe this from {start_time} to {end_time} seconds.',
        'Caption this from {start_time} to {end_time} seconds.',
        'What can be seen in this video from {start_time} to {end_time} seconds?',
        'What do you see in the video from {start_time} to {end_time} seconds?',
        'Look at this video carefully from {start_time} to {end_time} seconds and then tell me about it in detail',
        'Write a long description of this video from {start_time} to {end_time} seconds.',
        'Tell me about this video from {start_time} to {end_time} seconds.',
        'Write a paragraph about this video from {start_time} to {end_time} seconds.',
        'Generate a long caption about this video from {start_time} to {end_time} seconds.',
    ],
    "video_frame_caption_timestamp": [
        'Describe this video frame at {timestamp}.',
        'describe the video frame at {timestamp}.',
        'Write a long description of this video frame at {timestamp}.',
        'caption the video at {timestamp}.',
        'Caption the video frame at {timestamp}.',
        'caption the frame at {timestamp}.',
        'What can be seen in this video frame at {timestamp}?',
        'What do you see in the video frame at {timestamp}?',
        'Look at this video frame carefully at {timestamp} and then tell me about it in detail',
        'Write a long description of this video frame at {timestamp}.',
        'Tell me about this video frame at {timestamp}.',
        'Write a paragraph about this video frame at {timestamp}.',
        'Generate a long caption about this video frame at {timestamp}.',
    ],
    "video_frame_caption_timestamp_in_seconds": [
        'Describe this video frame at {timestamp} seconds.',
        'describe the video frame at {timestamp} seconds.',
        'Write a long description of this video frame at {timestamp} seconds.',
        'caption the video at {timestamp} seconds.',
        'Caption the video frame at {timestamp} seconds.',
        'caption the frame at {timestamp} seconds.',
        'What can be seen in this video frame at {timestamp} seconds?',
        'What do you see in the video frame at {timestamp} seconds?',
        'Look at this video frame carefully at {timestamp} seconds and then tell me about it in detail',
        'Write a long description of this video frame at {timestamp} seconds.',
        'Tell me about this video frame at {timestamp} seconds.',
        'Write a paragraph about this video frame at {timestamp} seconds.',
        'Generate a long caption about this video frame at {timestamp} seconds.',
    ],
    "video_clip_transcript_start_end": [
        'Describe this video as if you are a person speaking from {start_time} to {end_time}.',
        'Imagine you are a person talking about this video from {start_time} to {end_time}. Generate a transcript of what you would say.',
        "Generate an audio transcript of a person describing this video from {start_time} to {end_time}",
        "Create a transcript of a human describing this video out load from {start_time} to {end_time}",
        "Describe this from {start_time} to {end_time} in this style of a human talking",
    ],
    "video_clip_transcript_start_end_in_seconds": [
        'Describe this video as if you are a person speaking from {start_time} to {end_time} seconds.',
        'Imagine you are a person talking about this video from {start_time} to {end_time} seconds. Generate a transcript of what you would say.',
        "Generate an audio transcript of a person describing this video from {start_time} to {end_time} seconds",
        "Create a transcript of a human describing this video out load from {start_time} to {end_time} seconds",
        "Describe this from {start_time} to {end_time} seconds in this style of a human talking",
    ],
    "long_caption_no_pointing": [
        'Describe this image in detail, but without any pointing.',
        'Write a long description of this image, do not produce any points.',
        'Tell me about this picture, use plain text only.',
        'Generate a plain text description of this caption',
        "What is in this image?\nNo pointing\nGive lots of detail"
        "Write a long caption.\nDo not use image coordinates\nOutput a full paragraph"
    ],
    "transcript": [
        'Describe this image as if you are a person speaking',
        'Imagine you are a person talking about this image. Generate a transcript of what you would say.',
        "Generate an audio transcript of a person describing this image",
        "Create a transcript of a human describing this image out load",
        "Describe this in this style of a human talking",
    ],
    "refexp_pointing": [
        'Where is the \"{refexp}\"?',
        'Point to {refexp}',
        'point at {refexp}',
        'Find the {refexp}.',
        'Which object in the image does \"{refexp}\" refer to?',
        'Locate the object \"{refexp}\" refers to.',
        'Point to the object that best matches the expression:\n{refexp}\n',
        'What object could be described as: {refexp}.\nPoint:',
        'Referring Expression: {refexp}.\nPoint:',
        'Expression: {refexp}\nPoint to the refexp',
        'Task: Point to the object that best matches the expression.\nExpression: {refexp}\nPoint:',
        'Instruction: Locate the object that matches the expression by returning a point.\nReferring Expression: {refexp}\n',
        'Help me find an object in this image by pointing to the {refexp}',
        'What point of the image might the expression \'{refexp}\' refer to?',
    ],
    "plain": ["{question}"],
    "multiple_choice": [
        "{question}\n{options}\nReturn only the letter of the best answer option",
        "Answer this question by naming one of the provided options:\n{question}\n{options}",
        "{question}\n{options}\nWhat option best answers the question?",
        "{question}\n{options}\nReturn the best answer option",
        "Look at the options, then return the letter of the option that best answers the question.\nQuesiton: {question}\nOptions: {options}",
        "{question}? Select an answer option from:\n{options}",
        "{question}\nSelect an answer option from:\n{options}\n\n",
        "Question: {question}? Options: {options} Answer:",
        "Answer the question by selecting an answer options\nQuestion: {question}\nOptions: {options}",
        "{question}?\n{options}\nReturn only the letter of the correct answer",
        "Help me answer this question: \"{question}\", by stating which of the following options is correct\n{options}."
    ],
    "video_multiple_choice_w_subtitle": [
        "{question}\n{options}\nUse the subtitles as context if needed. Return only the letter of the best answer option.",
        "Answer this question by naming one of the provided options:\n{question}\n{options}\n(Subtitles are available for reference.)",
        "{question}\n{options}\nConsider the subtitles when relevant. What option best answers the question?",
        "{question}\n{options}\nUse the subtitles to guide your reasoning. Return the best answer option.",
        "Look at the options and subtitles, then return the letter of the option that best answers the question.\nQuestion: {question}\nOptions: {options}",
        "{question}? Select an answer option from:\n{options}\n(You may use the subtitles to help.)",
        "{question}\nSelect an answer option from:\n{options}\nUse the subtitles if needed for context.\n",
        "Question: {question}? Options: {options}\nSubtitles may contain useful context. Answer:",
        "Answer the question by selecting an answer option.\nQuestion: {question}\nOptions: {options}\n(Subtitles can provide supporting context.)",
        "{question}?\n{options}\nUse the subtitles if relevant. Return only the letter of the correct answer.",
        "Help me answer this question: \"{question}\", by stating which of the following options is correct\n{options}\nYou can reference the subtitles to decide."
    ],
    "video_multiple_choice_multiple_correct": [
        "{question}\n{options}\nReturn the best answer option(s). In the case of multiple correct options, separate them with newlines.",
        "{question}\n{options}\nWhat option(s) best answers the question? In the case of multiple correct options, separate them with newlines.",
        "{question}\n{options}\nReturn the best answer option(s). In the case of multiple correct options, separate them with newlines.",
        "Look at the options, then return the letter(s) of the option(s) that best answers the question.\nQuesiton: {question}\nOptions: {options}",
        "Question: {question}? Options: {options} Answer:",
    ],
    "pointing": [
        "Point to {label}\nPlease say 'There are none.' if it is not in the image.",
        "Point to all occurrences of \"{label}\"",
        "Point to any {label} in the image",
        "Point to any {label} in the image.",
        "Point: Where are the {label}",
        "Show me where the {label} are",
        "Can you show me where the {label} are?",
        "Show me where the {label} are",
        "Show me where a {label} is",
        "Show me where a {label} is.",
        "If there are any {label} in the image? Show me where they are.",
        "Where are the {label}?",
        "Generate a list of points showing where the {label} are.",
        "Find the \"{label}\".",
        "Find a \"{label}\".",
        "Locate all {label}.",
        "Locate an {label}.",
        "Locate a {label}.",
        "Locate every {label}.",
        "Locate {label}.",
        "Locate the {label}.",
        "Object: {label}\nInstruction: Point to the object.",
        "find {label}",
        "find {label}.",
        "Point to every {label}",
        "find any {label} in the picture",
        "Find the {label}",
        "Find any {label}",
        "Point to a {label}",
        "Point to an {label}",
        "Look for {label} in the image and show me where they are.",
        "Help me find an object in the image by pointing to them.\nObject: {label}.",
        "I am looking for {label}, where can they be found in the image?",
        "Can you see any {label} in the image? Point to them.",
        "Point out each {label} in the image.",
        "Point out every {label} in the image.",
        "Point to the {label} in the image.",
        "Locate each {label} in the image.",
        "Can you point out all {label} in this image?",
        "Please find {label} and show me where they are.",
        "If there are any {label} present, indicate their positions.",
        "If there is a {label} present, indicate its positions.",
        "show me all visible {label}",
    ],
    "point_count": [
        "How many {label} are there?",
        "How many {label}?",
        "How many {label}.",
        "how many {label}.",
        "how many {label}?",
        "How many \"{label}\" are there in the image?",
        "How many {label} are there in the image?",
        "Tell me how many {label} there are",
        "Tell me how many {label} there are and point to them.",
        "how many {label}",
        "Tell me where each {label} is.",
        "Tell me how many {label} are in the image",
        "count {label}",
        "count every {label}",
        "count each {label}",
        "count {label}.",
        "Count the {label}.",
        "How many {label} do you see?",
        "How many {label} are visible?",
        "Count all the {label}",
        "how mmny {label}?",
        "Count every {label} in the picture.",
        "Count all the {label}",
        "Count each {label}",
        "Point to and count the {label} in the picture.",
        "Point and count {label}",
        "Point to every {label}",
        "Locate the {label} and count them",
        "Locate every {label} and count them",
        "Find all the {label}. How many are there?",
        "Find each {label}. How many are there?",
        "Point at {label} and then tell me the count.",
        "What is the total number of {label} in the image?",
        "What is the number of {label}?",
        "In this image, how many {label} are there?",
        "In all the picture, how many {label} are there?",
        "Point at the {label} and then count them.",
        "Point to all the visible {label} output the total count.",
        "Point to all the {label} visible and output the total count. \nPlease say 'There are none.' if it is not in the image.",
        "Point to all occurrences of \"{label}\" and output the total count.",
        "Show me where the {label} are and output the total count.",
        "Where are the {label}? How many are there?",
        "Generate list of points showing where the {label} are and output the total count.",
        "Object: {label}\nInstruction: Point to the object and output the total count.",
        "find any {label} in the picture and output the total count.",
        "Can you see any {label} in the image? Point to them and output the total count.",
        "Can you point out all {label} in this image? How many are there?",
        "If there are any {label} present, indicate their positions and output the total count.",
        "How many {label} are there in the image? Point to them and output the total count.",
        "How many {label} are there in the image?",
        "Give me the count of {label} in the image.",
        "How many {label} are visible in the image?",
        "How many {label} are there?",
        "In the image, how many {label} are there?",
        "Can you count the number of {label} in the image?",
        "Can you count every {label} in the picture?",
        "Can you see any {label} in the image? How many are there?",
        "Are there any {label} in the image? How many are there?",
        "If you see any {label} in the image, give me the count. Otherwise, say 'There are none.'",
        "Object: {label}\nInstruction: How many are there?",
    ],
    "count_then_point": [
        "Count the {label} in the image, then point to them.",
        "How many {label} are there? Point to them.",
        "Count every {label} in the picture, then point to them.",
        "Locate the {label} and count them, then point to them.",
        "Find all the {label}. How many are there? Point to them.",
        "Find each {label}. How many are there? Point to them.",
    ],
    "only_count": [
        "Count the {label} in the image.",
        "How many {label} are there?",
        "Count every {label} in the picture.",
        "Locate the {label} and count them.",
        "Find all the {label}. How many are there?",
        "Find each {label}. How many are there?",
    ],
    "chain_of_thought": [
        "{question} Provide reasoning steps and then give the short answer.",
    ],
    "multi_image_pointing" : [
        "Find {selected_label} in {selected_images}.",
        "find {selected_label} in {selected_images}.",
        "Point to {selected_label} in {selected_images}.",
        "Point to any {selected_label} in {selected_images}.",
        "Point to all {selected_label} in {selected_images}.",
        "Point to all occurrences of \"{selected_label}\" in {selected_images}.",
        "Can you point to {selected_label} in {selected_images}?",
        "Show me where the {selected_label} are in {selected_images}?",
        "Show me where a {selected_label} is in {selected_images}?",
        "Show me where a {selected_label} is in {selected_images}.",
        "In {selected_images}, point to {selected_label}.",
        # "For each in {selected_images}, point to {selected_label}.",
        # "Mark {selected_label} in {selected_images}.",
        # "Pinpoint the location of {selected_label} in {selected_images}.",
        # "Annotate coordinates for each {selected_label} in {selected_images}.",
        # "Indicate all {selected_label} in {selected_images}.",
        # "For {selected_images}, mark every {selected_label} (provide coordinates per image).",
        # "Locate all {selected_label} in {selected_images} and list their points.",
        # "Tag all visible {selected_label} in {selected_images} with point coordinates.",
        # "Reveal the positions of all {selected_label} in {selected_images}.",
    ],
    "multi_image_point_then_count": [
        "How many {selected_label} are there in {selected_images}?",
        "how many {selected_label} are there in {selected_images}?",
        "How many {selected_label} are there in {selected_images}.",
        "count {selected_label} in {selected_images}?",
        "count every {selected_label} in {selected_images}?",
        "count each {selected_label} in {selected_images}?",
        "count {selected_label} in {selected_images}.",
        "Count the {selected_label} in {selected_images}.",
        "Point then count {selected_label} in {selected_images}.",
        "Point and count {selected_label} in {selected_images}.",
        "Can you point and count {selected_label} in {selected_images}?",
        "Can you point then count {selected_label} in {selected_images}?",
        "Point to {selected_label} in {selected_images}, then count them.",
        "Point to all {selected_label} in {selected_images}, then count them.",
        # "In {selected_images}, point to every {selected_label}, then count them.",
        # "For each in {selected_images}, show all {selected_label}, then provide the total count.",
        # "Mark all occurrences of {selected_label} in {selected_images}, then count how many there are.",
        # "Identify and indicate every {selected_label} visible in {selected_images}, then report the total number.",
        # "Highlight all instances of {selected_label} in {selected_images}, and afterward, give the count.",
        # "Pinpoint the location of each {selected_label} in {selected_images}, then state how many you found.",
        # "Outline every {selected_label} you see in {selected_images}, followed by the total count.",
        # "Annotate coordinates for each {selected_label} in {selected_images}, then sum up how many appear.",
        # "Indicate all {selected_label} in {selected_images}; return the points first, then the total count.",
        # "Show where every {selected_label} appears in {selected_images}, then report the total number.",
        # "For {selected_images}, mark every {selected_label} (provide coordinates per image), then give the overall count.",
        # "Locate all {selected_label} in {selected_images} and list their points, then provide the total number found.",
        # "Tag all visible {selected_label} in {selected_images} with point coordinates, then state the total count.",
        # "Reveal the positions of all {selected_label} in {selected_images}, then count them.",
    ],
    "multi_image_counting": [
        "Total {selected_label} across {selected_images}?",
        "How many {selected_label} are there in total across {selected_images}?",
        "Compute the sum of {selected_label} counts over {selected_images}.",
        "What is the combined number of {selected_label} across all images?",
        "Add up the {selected_label} across {selected_images}; whats the result?",
        "Provide the aggregate count of {selected_label} over {selected_images}.",
        "Across {selected_images}, how many {selected_label} in aggregate?",
        "Whats the total when you sum {selected_label} across {selected_images}?",
        "Give the grand total of {selected_label} across {selected_images}."
    ],
    "multi_image_count_then_point": [
        "Total {selected_label} across {selected_images}? Then point to the result.",
        "How many {selected_label} are there in total across {selected_images}? After answering, point to the total.",
        "Compute the sum of {selected_label} counts over {selected_images}, then point to the total count.",
        "What is the combined number of {selected_label} across all images? Afterwards, point to where the total applies.",
        "Add up the {selected_label} across {selected_images}; what's the result? Then point to the total.",
        "Provide the aggregate count of {selected_label} over {selected_images}, and point to that total after answering.",
        "Across {selected_images}, how many {selected_label} in aggregate? Then point to the total.",
        "What's the total when you sum {selected_label} across {selected_images}? After that, point to the corresponding location.",
        "Give the grand total of {selected_label} across {selected_images}, then point to the total afterward."
    ],
    "most_least_selected_image": [
        "Which image in {selected_images} contains the most {selected_label}?",
        "From {selected_images}, pick the image with the highest {selected_label} count.",
        "Identify the image in {selected_images} with the maximum number of {selected_label}.",
        "In {selected_images}, which image shows the greatest number of {selected_label}?",
        "Which image id leads in {selected_label} count within {selected_images}?",
        "Select the top image by {selected_label} count from {selected_images} (list all if tied).",
        "Which image in {selected_images} has the fewest {selected_label}?",
        "From {selected_images}, pick the image with the lowest {selected_label} count.",
        "Identify the image in {selected_images} with the minimum number of {selected_label}.",
        "In {selected_images}, which image shows the least {selected_label}?",
        "Which image id trails in {selected_label} count within {selected_images}?",
        "Select the bottom image by {selected_label} count from {selected_images} (list all if tied).",
    ],
    "rank_by_cnt": [
        "Rank {selected_images} from fewest to most {selected_label}.",
        "Order {selected_images} by {selected_label} count (ascending).",
        "Sort {selected_images} by number of {selected_label} (low to high).",
        "List {selected_images} in increasing order of {selected_label} frequency.",
        "Arrange {selected_images} from smallest to largest {selected_label} count.",
        "Rank {selected_images} from most to fewest {selected_label}.",
        "Order {selected_images} by {selected_label} count (descending).",
        "Sort {selected_images} by number of {selected_label} (high to low).",
        "List {selected_images} in decreasing order of {selected_label} frequency.",
        "Arrange {selected_images} from largest to smallest {selected_label} count.",
        "Provide a ranking of {selected_images} by {selected_label} count (ties may share a position).",
        "Return {selected_images} sorted by {selected_label} count; keep ties adjacent."
    ],
    "exact_cnt": [
        "Select all images with exactly {n} {selected_label}.",
        "Which images in {selected_images} contain precisely {n} {selected_label}?",
        "Find images where the number of {selected_label} equals {n}.",
        "Return the image ids that have {n} instances of {selected_label} (no more, no fewer).",
        "Choose all images showing exactly {n} occurrences of {selected_label}.",
        "Pick images in {selected_images} with an exact {selected_label} count of {n}.",
        "Identify images whose {selected_label} count is {n}.",
        "Mark the images matching count({selected_label}) = {n}.",
        "List images with {n} and only {n} {selected_label}.",
        "Which images satisfy an exact {selected_label} count of {n}?",
    ],
    "threshold_cnt": [
        "Select images with at least {n} {selected_label}.",
        "Which images have {n} or more {selected_label}?",
        "Return images with no fewer than {n} {selected_label}.",
        "Find images where count({selected_label}) greater than or equal to {n}.",
        "Pick every image whose {selected_label} count meets or exceeds {n}.",
        "Choose images showing a minimum of {n} {selected_label}.",
        "From {selected_images}, list those that contain {selected_label} in quantity greater than or equal to {n}.",
        "Identify images with {selected_label} count not less than {n}.",
        "Select all images that have at least {n} occurrences of {selected_label}.",
        "Which image ids satisfy count({selected_label}) greater than or equal to {n}."
    ],
}


GENERAL_PROMPTS_V1["video_short_caption"] = [prompt.replace("image", "video").replace("picture", "video").replace("photo", "video") for prompt in GENERAL_PROMPTS_V1["short_caption"]]
GENERAL_PROMPTS_V1["video_long_caption"] = [prompt.replace("image", "video").replace("picture", "video").replace("photo", "video") for prompt in GENERAL_PROMPTS_V1["long_caption"]]
GENERAL_PROMPTS_V1["video_transcript"] = [prompt.replace("image", "video") for prompt in GENERAL_PROMPTS_V1["transcript"]]
GENERAL_PROMPTS_V1["video_short_answer"] = [prompt.replace("image", "video") for prompt in GENERAL_PROMPTS_V1["short_answer"]]
GENERAL_PROMPTS_V1["video_multiple_choice"] = GENERAL_PROMPTS_V1["multiple_choice"]
#TODO: ablate over video multiple choice with subtitles: specific templates vs normal multiple choice templates
GENERAL_PROMPTS_V1["video_multiple_choice_w_subtitle"] =GENERAL_PROMPTS_V1["multiple_choice"]
GENERAL_PROMPTS_V1["video_point"] = [prompt.replace("image", "video").replace("picture", "video").replace("photo", "video") for prompt in GENERAL_PROMPTS_V1["pointing"]]
GENERAL_PROMPTS_V1["video_point_count"] = [prompt.replace("image", "video").replace("picture", "video").replace("photo", "video") for prompt in GENERAL_PROMPTS_V1["point_count"]]
GENERAL_PROMPTS_V1["video_count"] = [prompt.replace("image", "video").replace("picture", "video").replace("photo", "video") for prompt in GENERAL_PROMPTS_V1["only_count"]]
GENERAL_PROMPTS_V1["video_count_point"] = [prompt.replace("image", "video").replace("picture", "video").replace("photo", "video") for prompt in GENERAL_PROMPTS_V1["count_then_point"]]

# Constants for multi-image pointing tasks
NEGATIVE_LABELS = [
    "umbrella", "wallet", "key", "mirror", "clock", "candle", "towel", "blanket", "pillow",
    "drawer", "curtain", "cabinet", "toaster", "kettle", "microwave", "sink", "bathtub",
    "toilet", "bench", "mailbox", "hydrant", "traffic light", "skateboard", "sign", "bus stop",
    "goat", "duck", "rabbit", "squirrel", "frog", "snake", "bear", "monkey", "elephant", "tiger",
    "tie", "scarf", "belt", "bracelet", "necklace", "earrings", "boots", "sandals", "gloves",
    "pan", "cutting board", "spatula", "ladder", "broom", "mop", "bucket", "tape", "rope",
    "nail", "bolt", "flashlight", "drill", "hammer", "wrench", "pliers", "saw", "camera",
    "flower", "mountain", "river", "lake", "cloud", "moon", "star", "leaf", "sand", "rock",
    "router", "projector", "vacuum", "fan", "washing machine", "printer", "speaker", "headphone",
    "tablet", "poster", "map", "gift", "toy", "ball", "coin", "card", "pen drive", "wine", "beer",
    "sandwich", "salad", "pasta", "bread", "egg", "cheese", "cereal", "pot", "bowl", "plate",
    "toothbrush", "soap", "lamp", "plant", "curtain rod", "bookend", "shovel", "measuring cup",
    "thermos", "remote", "binoculars", "lantern", "stapler", "paintbrush", "trophy", "helmet",
    "mask", "compass", "telescope", "dice", "chessboard", "microphone", "record player", "harp",
    "guitar", "violin", "keyboard instrument", "drum", "trumpet", "skis", "snowboard", "surfboard",
    "tent", "campfire", "sleeping bag", "bin", "scale", "typewriter", "ink bottle", "measuring tape"
]


# Point-Tracking Prompts
GENERAL_PROMPTS_V1["video_point_track_per_frame_default_fps"] = [
    "Track {label}.",
    "track the {label}.",
    "track every {label}.",
    "track each {label}",
    "please track \"{label}\".",
    "Track: {label}",
    "tracking: {label}",
    " track the all {label}",
]
GENERAL_PROMPTS_V1["video_point_track_per_frame"] = [
    "Track {label} in {fps} FPS.",
    "Track {label} at {fps} FPS",
    "Track the {label} in {fps} FPS",
    "Track all instances of '{label}' in this video, sampling at {fps} frames per second. Show the position coordinates at each timestamp.",
    "Follow the {label} throughout this video. Sample positions at {fps} FPS and mark each with coordinates.",
    "Identify and track each {label} in this video clip. Sample at {fps} frames per second and provide point coordinates.",
    "Monitor the movement of all {label} in this video. Sample at {fps} FPS and output their positions as coordinates.",
    "For each {label} in the video, track its position throughout the clip sampling at {fps} FPS. Return coordinates at each sampled frame.",

    # html-v1 format
    "Track all instances of '{label}' in this video, sampling at {fps} frames per second. Show the position coordinates at each timestamp, given as <track coords=\"t id x y\">label</tracks>.",
    "For each {label} in the video, track its position throughout the clip sampling at {fps} FPS. Return coordinates at each sampled frame, given as <track coords=\"t id x y\">label</tracks>.",
]
GENERAL_PROMPTS_V1["video_point_ground_start_end"] = [
    "For each '{label}' in this video, identify only its first and last appearances with precise coordinates.",
    "Locate when each {label} first appears and last appears in the video. Provide exact coordinates for both moments.",
    "Find the coordinates of {label} at their entry and exit points in this video.",
    "Mark the positions of each {label} at its first and final appearances in the video.",
    "Show where each {label} enters and exits the frame with exact point coordinates.",

    # html-v1 format
    "For each '{label}' in this video, identify only its first and last appearances with precise coordinates, given as <track coords=\"t id x y\">label</tracks>.",
    "Show where each {label} enters and exits the frame with exact point coordinates, given as <track coords=\"t id x y\">label</tracks>.",
]
GENERAL_PROMPTS_V1["video_single_point_track_per_frame"] = [
    "Track {input_points} in {fps} FPS.",
    "{input_points} track in {fps} FPS.",
    "{input_points} Track the object in {fps} FPS.",
    
    # Adapt to html-v1 format
    'Track the following point {input_points} given as <points coords="t id x y">label</points>. Provide coordinates at {fps} frames per second',
    'Follow this point {input_points} in the video, given as <points coords="t id x y">label</points>. Sample at {fps} FPS and provide coordinates for each frame.',
    'For the point {input_points} in <points coords="t id x y">label</points> format, track its position in every frame of the video at {fps} FPS.',
    'Track the movement of the point {input_points} frame-by-frame at {fps} FPS, providing position coordinates for each frame.',
    'Can you track the point {input_points} format throughout the video at {fps} FPS?',
    'Starting from the point {input_points} in <points coords="t id x y">label</points> format, track its position at {fps} FPS.',
    'Sample at {fps} FPS and provide coordinates for the point {input_points}.',
    'Object tracking: {input_points} starting position, {fps} FPS rate.',
    'Object tracking: {input_points} formatted as <points coords="t id x y">label</points>, {fps} FPS rate.',
]

## End of Point-Tracking Prompts

STYLE_TO_GENERAL_PROMPT = {
    "vqa2": "short_answer",
    "coco_captioning": "short_caption",
    "gqa": "short_answer",
    "ocr_vqa": "short_answer",
    "tally_qa": "short_answer",
    "text_vqa": "short_answer",
    "okvqa": "short_answer",
    "chart_qa": "short_answer",
    "doc_qa": "short_answer",
    "info_qa": "short_answer",
    "science_qa": "multiple_choice",
    "ai2_diagram": "multiple_choice",
    "a_okvqa_mc": "multiple_choice",
    "a_okvqa_da": "short_answer",
    "long_caption": "long_caption",
    "scifi_charts": "short_answer",
    "scifi_charts_qa": "short_answer",
    "pointing": "pointing",
    "point_count": "point_count",
    "count_then_point": "count_then_point",
    "only_count": "only_count",
    "plain": "plain",
}


def seconds_to_timestamp(seconds, second_decimal_places=2):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60  # Keep decimal
    seconds = f"%0{second_decimal_places+3}.{second_decimal_places}f" % seconds
    formatted = f"{hours:02}:{minutes:02}:{seconds}"
    return formatted


def apply_keywords(prompt, example, keywords):
    for keyword in keywords:
        res = prompt.split("{"+keyword+"}", maxsplit=2)
        prompt = res[0] + example[keyword] + res[1]
    return prompt


def apply_keyword_prompt(prompts, example, rng, keywords=None, dbg=False):
    if isinstance(prompts, list):
        assert keywords is None
        all_keywords = [sorted(re.findall("{([^{}]+)}", x)) for x in prompts]
        keywords = all_keywords[0]
        assert len(keywords) == len(set(keywords)), f"Repeated keywords in {keywords}"
        assert all(keywords == x for x in all_keywords), f"Inconsistent keywords in prompts {all_keywords}"
        assert not any("{" not in word[1:-1] and "}" in word[1:-1] for word in keywords)

        for k in keywords:
            assert k in example, f"Example missing expected field {k}, example={example}"

    if dbg:
        prompt = prompts[0]
    else:
        prompt = prompts[rng.randint(0, len(prompts))]
    return apply_keywords(prompt, example, keywords)


DEMO_STYLES = [
    "point_count",
    "pointing",
    "cosyn_point",
    "user_qa",
    "long_caption",
    "short_caption",
    "video_long_caption",
    "video_short_caption",
    "video_point_track_per_frame",
    "video_point_track_start_end",
    # "video_point_track_all_frames",
    "video_single_point_track_start_end",
    "video_transcript",
    "video_clip_caption_start_end",
    "video_clip_caption_start_end_in_seconds",
    "video_clip_transcript_start_end",
    "video_clip_transcript_start_end_in_seconds",
    "video_frame_caption_timestamp",
    "video_frame_caption_timestamp_in_seconds",
    "correction_qa",
    "text_sft",
    "video_point",
    "video_point_count",
    "video_count",
    "video_count_point",
    "multi_image_pointing",
    "multi_image_counting",
    "multi_image_point_then_count",
    "multi_image_count_then_point",
]

VIDEO_MC_STYLES = [
    "video_multiple_choice",
    "video_multiple_choice_count_without_pointing",
    "video_multiple_choice_multiple_correct",
    "video_multiple_choice_w_subtitle"
]

IMAGE_MC_STYLES = [
    "a_okvqa_mc",
    "ai2_diagram_no_letter",
    "ai2_diagram",
    "science_qa",
    "eval_multiple_choice",
    'a_okvqa_mc',
    "multi_image_mc",
    "multi_image_mc_exp",
    "mantis_instruct_mc",
    "eval_multi_image_mc"
]


@dataclasses.dataclass
class DataFormatter(BaseConfig):
    """
    Applies prompt templates and adds system prompts to construct text inputs/output

    Includes methods for formatting points, video points, and annotated text.
    """
    prompt_templates: str = "none"  # How to template prompts for examples
    message_format: str = "none"  # How to format messages
    system_prompt: Optional[str] = None  # How to generate system prompts
    always_start_with_space: bool = False  # Always include a leading space for the first bit of text
    default_inference_len: Optional[int] = 65  # Inference len for length-conditioned prompting
    select_answer: str = "best"  # How to select answer for questions with many answers
    debug: bool = False  # deterministic mode for debugging
    image_last: bool = False
    format_message_list: Optional[str] = None
    p_one_message: float = 0
    eval_system_prompt_mapping: Optional[Dict[str, str]] = None # Custom mapping from eval system prompt styles to train styles
    p_choice_content_in_mc: float = 1.0
    template_video_mc_questions: bool = True
    pointing_format: str = "default"
    points_decimal_places: int = 1  # Decimal places for points in text
    use_seperate_non_pointing_qa_style: bool = False
    timestamp_mode: str = "50-percent-seconds"
    output_timestamp_mode: str = "seconds"
    seconds_decimal_places: int = 1
    p_multi_point_all_image: float = 0

    use_seperate_count_without_pointing_style: bool = False

    _point_formatter: Optional[PointFormatter] = dataclasses.field(default=None, metadata={"omegaconf_ignore": True})
    sample_random_initial_point: bool = True  # For video point tracking, whether to sample random initial point

    def __post_init__(self):
        if self.pointing_format in ["html-v1", "html-v2"]:
            self._point_formatter = UnifiedPointFormatter.build_for_format(self.pointing_format, self.format_output_timestamp)
        else:
            self._point_formatter = LegacyPointFormatting(self.pointing_format, self.seconds_decimal_places)
        if self.prompt_templates == "uber_model_v2":
            assert self.system_prompt != "demo_or_style"

    def _get_scale(self, example):
        """Get scale factor from example."""
        # FIXME need to figure out what this does for multi-image pointing
        if "point_scale" in example:
            # Points are already normalized
            scale = example["point_scale"]
            return [scale, scale]
        elif "image" in example and isinstance(example["image"], np.ndarray):
            h, w = example["image"].shape[:2]
            return [w, h]
        else:
            # Points are in pixel coordinate
            w = example.get("width", example.get("w"))
            h = example.get("height", example.get("h"))
            return [w, h]

    def _select_normalized_human_readable_label(self, example: Dict, rng) -> Tuple[str, str]:
        """Select a normalized label and find its human-readable equivalent."""
        # Get unique normalized labels and select one
        unique_labels = list(set(example["normalized_labels"]))
        selected_normalized_label = rng.choice(unique_labels)

        # Find the original human-readable label for this normalized label
        selected_label = self._find_human_readable_label(example, selected_normalized_label)

        return selected_normalized_label, selected_label

    def _find_human_readable_label(self, example: Dict, normalized_label: str) -> str:
        """Find the original human-readable label for a normalized label."""
        for i, norm_label in enumerate(example["normalized_labels"]):
            if norm_label == normalized_label:
                return example["labels"][i]
        return normalized_label  # fallback

    def format_annotated_text(self, answer, point_annotations):
        for point_annotation in point_annotations:
            parts = answer.split("<|POINT|>", maxsplit=1)
            # this is hard-coded for point-qa, which needs to be clipped and
            # has a scale of 0 to 100
            points = np.array(point_annotation["points"])
            points = np.clip(points, 0, 100)
            point_text = self._point_formatter.format_image_points(
                points, 100,
                point_annotation["inline_text"], point_annotation["alt_text"],
                mode="point"
            )
            answer = parts[0] + point_text + parts[1]
        return answer

    def template_options(self, example, is_training, rng):
        """Template a multiple choice questions"""
        labelled_options = "options" in example
        allow_unlabelled = True
        if labelled_options and "answer_idx" in example:
            idx = example["answer_idx"]
            if isinstance(idx, int):
                # If the correct answer is a blank make sure we label the options so there is
                # something sensible to output
                allow_unlabelled = bool(str(example["options"][idx]).strip())

        if not is_training or rng.random() < 0.1:
            # "Standard case that we default to during eval
            # We oversample this case to 100% sure its well covered in the training data
            if labelled_options:
                prefixes = string.ascii_uppercase
                options = example["options"]
                option_text = "\n".join(f"{prefix}. {opt}" for prefix, opt in zip(prefixes, options))
                option_names = prefixes[:len(options)]
                outputs = [
                    f"{name}. {opt}" for name, opt in zip(option_names, options)
                ]
            else:
                options = example["unlabelled_options"]
                option_text = "\n".join(options)
                outputs = options
                option_names = options
            question = example["question"] + "\nOnly return the correct answer option.\n" + option_text
        else:
            question = example["question"]

            if labelled_options:
                options = example["options"]
            else:
                options = example["unlabelled_options"]
            question, option_names, outputs = template_mc_question(
                question, options, rng,
                p_label_options=0.8 if allow_unlabelled else 1.0

            )

        if "answer_idx" in example:
            ans_idx = example["answer_idx"]
            if not (0 <= ans_idx < len(option_names)):
                raise ValueError(f"Invalid answer idx in example: {example}")
            output = outputs[ans_idx]
        else:
            output = None
        return question, output, dict(option_names=option_names)

    def format_options(self, example):
        if "options" in example:
            prefixes = string.ascii_uppercase
            options = example["options"]
            option_text = "\n".join(f"{prefix}. {opt}" for prefix, opt in zip(prefixes, options))
            option_names = prefixes[:len(options)]
        else:
            options = example["unlabelled_options"]
            option_text = "\n".join(options)
            prefixes = options
            option_names = options
        if "answer_idx" in example:
            ans_idx = example["answer_idx"]
            p = random.random()
            if "options" in example and p < self.p_choice_content_in_mc and example.get("content_in_mc", True):
                output = f"{prefixes[ans_idx]}. {options[ans_idx]}"
            else:
                output = prefixes[ans_idx]
        else:
            output = None
        return output, example["question"] + "\n" + option_text + "\n", dict(option_names=option_names)

    def format_points(self, example):
        if "points" not in example:
            return None
        points = example["points"]
        style = example["style"]
        if "label" in example:
            label = example["label"].lower()
        else:
            label = example["question"]
        if len(points) == 0:
            if style in ["pointing", "point_count"]:
                return "There are none."
            else:
                raise NotImplementedError()
        scale = self._get_scale(example)
        if example.get("clip_points"):
            points = np.maximum(points, 0)
            points = np.minimum(points, np.array(scale)[None, :])
        mode = style
        if style == "cosyn_point":
            mode = "point"
        return self._point_formatter.format_image_points(points, scale, label, mode=mode)

    def format_multi_points(self, example):
        """Format multi-image pointing and counting.

        Generates questions and answers for pointing to and counting objects across multiple images.
        Supports various modes: pointing only, counting only, or combined operations.
        """

        def point_count(example, rng):
            """Point and Count in randomized order to specific object(s) in the images with modified selection logic"""
            # Get unique normalized labels and select one
            selected_normalized_label, selected_label = self._select_normalized_human_readable_label(example, rng)

            # about 10% of the time select a negative label.
            # if rng.random() < 0.1:
            #     selected_normalized_label = rng.choice(NEGATIVE_LABELS)
            #     selected_label = selected_normalized_label  # negative labels are already human-readable

            # Use ALL images in the example, not just valid ones
            all_images = list(range(len(example["normalized_labels"])))
            n_images = len(all_images)

            # Check if the selected label exists in any of the images
            label_exists = False
            for i, (label, points) in enumerate(zip(example["normalized_labels"], example["points"])):
                if label == selected_normalized_label and len(points) > 0:
                    label_exists = True
                    break

            if self.p_multi_point_all_image:
                if n_images == 1 or rng.random() < self.p_multi_point_all_image:
                    selected_images = "all images"
                else:
                    n_images_to_select = rng.randint(1, n_images)
                    selected_images = rng.sample(all_images, n_images_to_select)
                    selected_images = ", ".join([f"image_{i+1}" for i in selected_images])
            else:
                # Randomly select from 1 to total number of available images plus "all images" option
                n_images_to_select = rng.randint(1, n_images) if n_images >= 1 else n_images
                selected_images = rng.sample(all_images, n_images_to_select)

                # Randomly select between "all images" or specific image list
                if n_images_to_select == n_images and rng.random() < 0.5:
                    selected_images = "all images"
                else:
                    selected_images = ", ".join([f"image_{i+1}" for i in selected_images])

            style = example.get("style")
            if selected_images == "all images" and style == "multi_image_pointing":
                # 50% chance to use original pointing template for all images
                if rng.random() < 0.5:
                    prompt_template = rng.choice(GENERAL_PROMPTS_V1["pointing"])
                    question = prompt_template.format(label=selected_label)
                else:
                    prompt_template = rng.choice(GENERAL_PROMPTS_V1[style])
                    question = prompt_template.format(
                        selected_images=selected_images,
                        selected_label=selected_label
                    )

                # NOTE: when "all images", we can always consider Qs without selected images.
            else:
                prompt_template = rng.choice(GENERAL_PROMPTS_V1[style])
                question = prompt_template.format(
                    selected_images=selected_images,
                    selected_label=selected_label
                )

            # Determine mode based on style
            if style == "multi_image_pointing":
                mode = "point"
            elif style == "multi_image_counting":
                mode = "count"
            elif style == "multi_image_point_then_count":
                mode = "point_then_count"
            elif style == "multi_image_count_then_point":
                mode = "count_then_point"

            if not label_exists:
                answer = f"There are none."
            else:
                # Find images that have this label with points for the answer
                # Only consider the selected images, not all images
                if selected_images == "all images":
                    valid_images = []
                    for i, (label, points) in enumerate(zip
                        (example["normalized_labels"], example["points"])):
                        if label == selected_normalized_label and len(points) > 0:
                            valid_images.append(i)
                else:
                    # Convert "image_1, image_2" to [0, 1]
                    selected_indices = [int(img.split('_')[1]) - 1
                        for img in selected_images.split(', ')]
                    valid_images = []
                    for i in selected_indices:
                        if (i < len(example["normalized_labels"]) and
                            example["normalized_labels"][i] == selected_normalized_label and
                            len(example["points"][i]) > 0):
                            valid_images.append(i)

                if not valid_images:
                    answer = f"There are none."
                else:
                    # Prep for format_multi_image_points
                    image_indices = []
                    points_list = []
                    scale_list = []
                    for i in valid_images:
                        points = example["points"][i]
                        scale_list.append(self._get_scale(example))
                        if points:
                            # Convert points from dict format {'x': x, 'y': y} to tuple format (x, y)
                            # handle very few negative values in the points
                            converted_points = []
                            for point in points:
                                if example.get("clip_points"):
                                    scale = example['point_scale']
                                    if isinstance(point, dict) and 'x' in point and 'y' in point:
                                        x, y = max(0, min(point['x'], scale)), max(0, min(point['y'], scale))
                                        converted_points.append((x, y))
                                    elif isinstance(point, (list, tuple)) and len(point) == 2:
                                        x, y = max(0, min(point[0], scale)), max(0, min(point[1], scale))
                                        converted_points.append((x, y))
                                else:
                                    converted_points.append(point)

                            image_indices.append(i + 1)  # Convert to 1-based indexing
                            points_list.append(converted_points)

                    if points_list:
                        # Get scales for the selected images in the same order as points_list
                        answer = self._point_formatter.format_multi_image_points(
                            image_indices=image_indices,
                            points=points_list,
                            scale=scale_list,
                            label=selected_label,
                            mode=mode
                        )
                    else:
                        answer = f"There are none."

            return question, answer

        def img_most_least_cnt(example, rng):
            """Find image with most/least count of objects"""
            # Get unique normalized labels and select one
            selected_normalized_label, selected_label = self._select_normalized_human_readable_label(example, rng)

            # Find counts for this specific label
            label_counts = []
            for i, (label, count) in enumerate(zip(example["normalized_labels"], example["counts"])):
                if label == selected_normalized_label:
                    label_counts.append((i, count))

            if not label_counts:
                # Fallback to all counts
                label_counts = [(i, count) for i, count in enumerate(example["counts"])]

            # Format selected_images as "image_1, image_2, ..."
            selected_images = ", ".join([f"image_{i+1}" for i, _ in label_counts])

            most_least = "most" if rng.random() > 0.5 else "least"

            if most_least == "most":
                max_count = max(label_counts, key=lambda x: x[1])[1]
                target_indices = [idx for idx, count in label_counts if count == max_count]
                # Use prompt template for most
                prompt_template = rng.choice(GENERAL_PROMPTS_V1["most_least_selected_image"][:6])  # First 6 are for "most"
            else:
                min_count = min(label_counts, key=lambda x: x[1])[1]
                target_indices = [idx for idx, count in label_counts if count == min_count]
                # Use prompt template for least
                prompt_template = rng.choice(GENERAL_PROMPTS_V1["most_least_selected_image"][6:])  # Last 6 are for "least"

            question = prompt_template.format(selected_images=selected_images, selected_label=selected_label)
            answer = f"[{', '.join([f'image_{idx + 1}' for idx in target_indices])}]"

            return question, answer

        def rank_by_cnt(example, rng):
            """Rank images by object count"""
            # Get unique normalized labels and select one
            selected_normalized_label, selected_label = self._select_normalized_human_readable_label(example, rng)

            # Create ranking based on counts for this label
            image_rankings = []
            for i, (label, count) in enumerate(zip(example["normalized_labels"], example["counts"])):
                if label == selected_normalized_label:
                    image_rankings.append((i, count))

            if not image_rankings:
                # Fallback to all images
                raise ValueError(f"No images contain {selected_label} objects to perform ranking.")
            elif len(image_rankings) == 1:
                # If only 1 image has the label, randomly include other images with 0 count
                total_images = len(example["counts"])
                remaining_indices = [i for i in range(total_images) if i != image_rankings[0][0]]
                n_additional = total_images - 1

                # Randomly select n additional images
                if len(remaining_indices) >= n_additional:
                    additional_indices = rng.sample(remaining_indices, n_additional)
                else:
                    additional_indices = remaining_indices

                # Add the additional images with 0 count to image_rankings
                for idx in additional_indices:
                    image_rankings.append((idx, 0))

            # Format selected_images as "image_1, image_2, ..."
            selected_images = ", ".join([f"image_{i+1}" for i, _ in image_rankings])

            # Use prompt template
            prompt_template = rng.choice(GENERAL_PROMPTS_V1["rank_by_cnt"])
            question = prompt_template.format(selected_images=selected_images, selected_label=selected_label)

            # Determine if the question asks for ascending or descending order
            ascending_keywords = ["fewest to most", "ascending", "low to high", "increasing", "smallest to largest"]
            descending_keywords = ["most to fewest", "descending", "high to low", "decreasing", "largest to smallest"]

            is_ascending = any(keyword in question.lower() for keyword in ascending_keywords)
            is_descending = any(keyword in question.lower() for keyword in descending_keywords)

            # Sort by count based on the question type
            if is_ascending:
                image_rankings.sort(key=lambda x: x[1], reverse=False)  # ascending order
            elif is_descending:
                image_rankings.sort(key=lambda x: x[1], reverse=True)   # descending order
            else:
                # Default to descending if we can't determine
                image_rankings.sort(key=lambda x: x[1], reverse=True)

            ranked_indices = [f"image_{idx + 1}" for idx, _ in image_rankings]
            answer = f"[{', '.join(ranked_indices)}]"

            return question, answer

        def exact_cnt(example, rng):
            """Find images with exact count"""
            # Get unique normalized labels and select one
            selected_normalized_label, selected_label = self._select_normalized_human_readable_label(example, rng)

            # Find actual counts for this label
            label_counts = [count for label, count in zip(example["normalized_labels"], example["counts"])
                        if label == selected_normalized_label]

            if not label_counts:
                # Fallback to all counts
                label_counts = example["counts"]

            # Choose a target count that actually exists
            target_count = rng.choice(label_counts) if label_counts else rng.randint(1, 5)

            # Find images with this exact count
            matching_images = []
            for i, (label, count) in enumerate(zip(example["normalized_labels"], example["counts"])):
                if label == selected_normalized_label and count == target_count:
                    matching_images.append(i)

            if not matching_images:
                # Fallback to any image with this count
                for i, count in enumerate(example["counts"]):
                    if count == target_count:
                        matching_images.append(i)

            # Format selected_images as "image_1, image_2, ..."
            all_images = list(range(len(example["counts"])))
            selected_images = ", ".join([f"image_{i+1}" for i in all_images])

            # Use prompt template
            prompt_template = rng.choice(GENERAL_PROMPTS_V1["exact_cnt"])
            question = prompt_template.format(selected_images=selected_images, selected_label=selected_label, n=target_count)

            if matching_images:
                image_indices = [f"image_{idx + 1}" for idx in matching_images]
                answer = f"[{', '.join(image_indices)}]"
            else:
                answer = "[]"

            return question, answer

        def threshold_cnt(example, rng):
            """Find images with count above threshold"""
            # Get unique normalized labels and select one
            selected_normalized_label, selected_label = self._select_normalized_human_readable_label(example, rng)

            # Find actual counts for this label
            label_counts = [count for label, count in zip(example["normalized_labels"], example["counts"])
                        if label == selected_normalized_label]

            if not label_counts:
                # Fallback to all counts
                label_counts = example["counts"]

            # Choose a threshold that makes sense based on actual data
            max_count = max(label_counts) if label_counts else 0
            threshold = rng.randint(1, max(max_count, 2))

            # Find images with count >= threshold
            matching_images = []
            for i, (label, count) in enumerate(zip(example["normalized_labels"], example["counts"])):
                if label == selected_normalized_label and count >= threshold:
                    matching_images.append(i)

            if not matching_images:
                # Fallback to any image with this threshold
                for i, count in enumerate(example["counts"]):
                    if count >= threshold:
                        matching_images.append(i)

            # Format selected_images as "image_1, image_2, ..."
            all_images = list(range(len(example["counts"])))
            selected_images = ", ".join([f"image_{i+1}" for i in all_images])

            # Use prompt template
            prompt_template = rng.choice(GENERAL_PROMPTS_V1["threshold_cnt"])
            question = prompt_template.format(selected_images=selected_images, selected_label=selected_label, n=threshold)

            if matching_images:
                image_indices = [f"image_{idx + 1}" for idx in matching_images]
                answer = f"[{', '.join(image_indices)}]"
            else:
                answer = "[]"

            return question, answer

        # Store category functions in a dictionary
        qa_categories = {
            "multi_image_pointing_counting": point_count,
            # "most_least_selected_image": img_most_least_cnt,
            # "rank_by_cnt": rank_by_cnt,
            # "exact_cnt": exact_cnt,
            # "threshold_cnt": threshold_cnt,
        }

        category_names = list(qa_categories.keys())
        weights = [1 if name == "multi_image_pointing_counting" else 0 for name in category_names]
        selected_category = random.choices(category_names, weights=weights)[0]
        category_function = qa_categories[selected_category]

        question, answer = category_function(example, random.Random())

        return question, answer

    def format_video_points(self, example):
        """
        Format video points for counting objects across frames.
        """
        if "points" not in example or "timestamps" not in example:
            return None
        if "count" in example and example["count"] == 0:
            return "There are none."
        if "unanswerable" in example and example["unanswerable"]:
            return example["explanation"] if len(example["explanation"]) > 0 else f"Sorry, I can't count {example['label']}."
        all_points = example["points"]
        all_timestamps = example["timestamps"]
        style = example["style"]
        assert style.startswith("video_")
        mode = style[6:]
        all_points = [[[p["x"], p["y"]] for p in fr] for fr in all_points]
        return self._point_formatter.format_video_points(all_timestamps, all_points, 100, example["label"], mode=mode)

    def format_video_input_points(self, initial_points, scale) -> str:
        """
        Legacy method for formatting initial input points for user prompt.
        """
        point_id_mapping = {p['id']: i for i,p in enumerate(initial_points)}

        formatted = {}
        for p in initial_points:
            point = self._point_formatter._normalize_and_round_points(p["point"], scale)
            time_str = self.format_input_timestamps(None, p["time"])
            formatted[point_id_mapping[p['id']]] = [point[0], point[1], time_str]

        if len(formatted) == 0:
            return "There are none."

        pairs = [f"{k}: [{v[0]}, {v[1]}, {v[2]}]" for k, v in formatted.items()]
        return "{" + ", ".join(pairs) + "}"

    def _find_initial_points(self, frames_data) -> list:
        """
        Find initial points from their first visible frame.
        Used for findinig inital query points to track for point tracking

        Returns:
            List of dicts with keys: id, point [x,y], time, frame
        """
        initial_points = {}
        for frame_data in frames_data:
            for point_id, point_info in frame_data["points"].items():
                if point_id not in initial_points:
                    if not point_info.get("occluded", False):
                        initial_points[point_id] = {
                            'id': point_id,
                            'point': point_info["point"],
                            'time': frame_data["time"],
                            'frame': frame_data["frame"]
                        }
        initial_points = list(initial_points.values())
        initial_points.sort(key=lambda x: x['frame']*10000 + x['point'][0]*100 + x['point'][1]) # Sort by frame, then x, then y

        return initial_points

    def _sample_initial_point(self, frames_data, point_id, is_training):
        """
        Sample a single initial point from the first visible frame for the given point_id.
        Used for finding initial query point to track for single point tracking.
        If training, sample randomly among visible points.

        Returns:
            Dict with keys: id, point [x,y], time, frame
        """

        # Sample randomly among visible points
        visible_points = []
        for frame_data in frames_data:
            for point_id, point_info in frame_data["points"].items():
                if point_id == point_id:
                    if not point_info.get("occluded", False):
                        visible_points.append({
                            'id': point_id,
                            'point': point_info["point"],
                            'time': frame_data["time"],
                            'frame': frame_data["frame"]
                        })
        
        if not visible_points:
            return None
        
        if is_training and self.sample_random_initial_point: # sample beginning or any visible point
            return random.choice(visible_points) if random.random() < 0.2 else visible_points[0]
        else: # pick the first visible point for eval
            return visible_points[0]

    def _filter_frames_to_video(self, frames_data, video_timestamps, eps=1e-2):
        """
        Filter frames_data to only include frames that match actual video timestamps.
        Uses numpy broadcasting for efficient comparison.
        """
        if not frames_data or video_timestamps is None or len(video_timestamps) == 0:
            return []

        # Extract frame times from frames_data
        frame_times = np.array([parse_timestamp(f["time"]) for f in frames_data])

        # Compute difference matrix: (n_frames, n_video_timestamps)
        video_timestamps = np.array(video_timestamps)
        diff_matrix = np.abs(frame_times[:, None] - video_timestamps)

        # Find minimum difference for each frame and corresponding video index
        min_diffs = np.min(diff_matrix, axis=1)
        closest_indices = np.argmin(diff_matrix, axis=1)

        # Filter frames that have a match within epsilon
        filtered_frames = []
        for i, frame_data in enumerate(frames_data):
            if min_diffs[i] < eps:
                filtered_frame = dict(frame_data)
                # filtered_frame["frame"] = int(closest_indices[i])
                # filtered_frame["time"] = float(video_timestamps[closest_indices[i]])
                filtered_frames.append(filtered_frame)

        return filtered_frames

    def _sample_at_fps(self, frames_data, sampling_fps):
        """
        Sample frames at specified fps interval.
        Trick: generate timestamp grids based on sampling_fps, then use _filter_frames_to_video to algin frames.
        """
        if not frames_data or sampling_fps <= 0:
            return frames_data

        sampling_interval = 1.0 / sampling_fps

        # Generate target times on the sampling grid
        start_time = parse_timestamp(frames_data[0]["time"])
        end_time = parse_timestamp(frames_data[-1]["time"])

        # Align to grid: find first grid point >= start_time
        first_grid_point = np.ceil(start_time / sampling_interval) * sampling_interval

        # Generate grid points
        target_times = np.arange(first_grid_point, end_time + 1e-6, sampling_interval)

        # Use filter_frames_to_video to find closest frames to these target times
        return self._filter_frames_to_video(frames_data, target_times)

    def format_video_object_track_points(self, example, is_training=False, rng=None):
        """
        Format video points for tracking objects across frames.
        Keep only frames that match actual video timestamps.
        Sample frames at specified sampling fps for per-frame tracking.
        """
        style = example["style"]
        label = example["label"]
        sampling_fps = example["sampling_fps"]
        input_points = None
        scale = self._get_scale(example)

        if "points" not in example or not example["points"]:
            prompt_keywords = dict(label=label)
            if sampling_fps and sampling_fps > 0:
                prompt_keywords["fps"] = str(int(sampling_fps))
            prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1[style], prompt_keywords, rng, dbg=self.debug)
            return prompt, "There are none."

        # Get actual video timestamps if available
        video_info = example.get("video", {})
        timestamps = video_info.get("timestamps", None)

        # Filter frames to match actual video timestamps
        frames_data = example["points"]
        frames_data = self._filter_frames_to_video(frames_data, timestamps)
        # NOTE: if frames_data is empty after filtering, we still proceed to sample initial points from original frames later
        # Output will be just "There are none." in that case

        # Get ouptut
        if style == "video_point_track_per_frame":
            # Apply fps sampling if specified
            if sampling_fps and sampling_fps > 0:
                frames_data = self._sample_at_fps(frames_data, sampling_fps)
            output = self._point_formatter.format_video_tracks(frames_data, scale, label)

        elif style == "video_point_ground_start_end":
            # For ground_start_end, just use filtered frames without fps sampling
            output = self._point_formatter.format_video_tracks(frames_data, scale, label, start_end_only=True)

        elif style == "video_single_point_track_per_frame":
            # Sample intial point from first visible frame or randomly if training
            initial_points = example.get("initial_points")
            if not initial_points:
                # For single point tracking, we assume tracking only one point with id=0
                initial_point = self._sample_initial_point(frames_data, point_id=0, is_training=is_training)
                if initial_point is None: # no visible initial point found after filtering, so randomly pick one from original frames
                    print("No visible initial point found after filtering frames to video timestamps for single point tracking, sampling from original frames instead.")
                    initial_point = self._sample_initial_point(example["points"], point_id=0, is_training=is_training)
                assert initial_point is not None, "No visible initial point found for single point tracking"
                initial_points = [initial_point]

            sampling_fps = example.get("sampling_fps")
            if sampling_fps and sampling_fps > 0:
                frames_data = self._sample_at_fps(frames_data, sampling_fps)
            
            # Design prompt for input points
            if isinstance(self._point_formatter, UnifiedPointFormatter):
                assert sorted(x["id"] for x in initial_points) == list(range(len(initial_points)))
                input_points = self._point_formatter.format_video_points(
                    [x['time'] for x in initial_points],
                    [[x['point']] for x in initial_points],
                    scale,
                    label=example["label"],
                    mode=None
                )
            else:
                input_points: str = self.format_video_input_points(initial_points, scale)
            
            output = self._point_formatter.format_video_tracks(frames_data, scale, label,
                                                             single_point_track=True,
                                                             from_initial_points=initial_points)
        else:
            raise NotImplementedError(f"Unsupported video point style: {style}")
        
        
        # assert len(frames_data) > 0, "No frames left after filtering/sampling"

        if False and "question" in example:
            prompt = example["question"]
        else:
            prompt_keywords = dict(label=label)
            if sampling_fps and sampling_fps > 0:
                prompt_keywords["fps"] = str(int(sampling_fps))
            if input_points is not None:
                prompt_keywords["input_points"] = input_points
            if style == "video_point_track_per_frame" and prompt_keywords["fps"] == '2' and rng.random() < 0.5:
                del prompt_keywords["fps"]
                prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1["video_point_track_per_frame_default_fps"], prompt_keywords, rng, dbg=self.debug)
            else:
                prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1[style], prompt_keywords, rng, dbg=self.debug)
        return prompt, output

    def format_video_point_track_points(self, example, initial_points):
        """
        Format video points for tracking points across frames

        Input:
            example: dict with keys: style, points (list of dicts with keys: frame, time, points)
            initial_points: dict of initial points with keys: id, point [x,y], frame

        Output formats by style:
        - video_point_track_all_frames_with_occlusion: "time {t}\n{id: [x, y, occluded], ...}"
        """
        if "points" not in example or not example["points"]:
            return "There are none."

        style = example["style"]
        frames_data = example["points"]
        scale = self._get_scale(example)
        return self._point_formatter.format_video_tracks(frames_data, scale, example["label"], from_initial_points=initial_points)

    def select_vqa_answer(self, answers, rng):
        if answers is None or isinstance(answers, str):
            return answers
        if self.select_answer == "first":
            return min(answers)
        if self.select_answer == "best":
            counts = Counter(answers)
            max_count = max(counts.values())
            candidates = [k for k, v in counts.items() if v == max_count]
            return candidates[rng.randint(0, len(candidates))]
        else:
            raise NotImplementedError(self.select_answer)

    def format_messages(self, messages):
        """Applies system formatting to ith message from a sequence of messages"""
        out = []
        for ix, message in enumerate(messages):
            is_user = ix % 2 == 0
            if self.message_format == "qwen3":
                if is_user:
                    if ix != 0:
                        prefix = "<|im_end|>\n"
                    else:
                        prefix = ""
                    message = f"{prefix}<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
            else:
                if self.message_format == "none" or self.message_format is None:
                    pass
                elif self.message_format == "role":
                    if is_user:
                        message = "User: " + message + " Assistant:"
                else:
                    raise NotImplementedError(self.message_format)

                if ix != 0 or self.always_start_with_space:
                    message = " " + message
            out.append(message)
        return out

    def get_system_prompt(self, style, for_inference, messages, is_training, rng):
        # For eval only dataset
        if self.eval_system_prompt_mapping is not None and style in self.eval_system_prompt_mapping:
            # Prioritize eval system prompt mapping if provided
            style = self.eval_system_prompt_mapping[style]
        else:
            # For eval, or if use_seperate_non_pointing_qa_style has been turned off,
            # remove the "count_without_pointing" style
            if (not is_training) or (not self.use_seperate_count_without_pointing_style):
                if style == "video_short_answer_count_without_pointing":
                    style = "video_short_answer"
                if style == "video_multiple_choice_count_without_pointing":
                    style = "video_multiple_choice"

            if style == "eval_short_answer":
                style = "vqa2"
            elif style == "eval_multiple_choice":
                style = "a_okvqa_mc"
            elif style == "video_eval_short_answer":
                style = "video_short_answer"
            elif style == "video_eval_multiple_choice":
                style = "video_multiple_choice"
            elif style == "video_eval_multiple_choice_w_subtitle":
                style = "video_multiple_choice_w_subtitle"
            elif style.startswith("eval_multi_image_"):
                style = style[len("eval_"):]

        if self.system_prompt == "style":
            return style + ":"

        elif self.system_prompt == "demo_or_style":
            if style == "android_control" or style == "demo":
                # android is a special case since I hacked in prefix in the preprocessor
                prefix = ""
            elif style in DEMO_STYLES and rng.random() > 0.1 and not self.debug:
                # Use style prompt 10% of the time so we can still get task-specific output
                prefix = ""
            else:
                prefix = style + ":"

        elif self.system_prompt == "demo_or_style_v2":
            # not percent chance to style use the style tag, all MC questions do not get a
            # style tag
            if style in DEMO_STYLES or style in VIDEO_MC_STYLES or style in IMAGE_MC_STYLES:
                prefix = ""
            else:
                prefix = style + ":"

        elif self.system_prompt in ["style_and_length", "style_and_length_v2"] and (
            style in ["pointing", "point_count", "cosyn_point"]):
            prefix = style + ":"

        elif for_inference and self.system_prompt in ["style_and_length", "style_and_length_v2"]:
            v2 = self.system_prompt == "style_and_length_v2"
            inference_len = self.default_inference_len
            n = None if inference_len is None else str(inference_len)
            if n is not None and len(n) > 0:  # allow empty string to signal unconditioned
                prefix = style + " " + n + ":"
            else:
                if self.system_prompt in ["style_and_length_v2"]:
                    prefix = style + ":"
                else:
                    prefix = style + " :"
        elif self.system_prompt in ["style_and_length_v3"]:
            # Length hint noise based on a percent of the total length instead of a staticly
            # defined factor
            if for_inference:
                n = self.default_inference_len
            elif rng.random() > 0.10:
                n = len(messages[-1])
                n *= np.clip(rng.normal(scale=0.05, loc=1), 0.5, 1.5)
                n = int(n / 25)
            else:
                n = None
            if n is not None:
                prefix = style + " " + str(n) + ":"
            else:
                prefix = style + ":"
        elif self.system_prompt in ["style_and_length", "style_and_length_v2"]:
            std = 25
            if rng.random() > 0.10:
                n = len(messages[-1])
                n += int(rng.normal(scale=std))
                n = n // 15
            else:
                n = None
            if n is not None:
                prefix = style + " " + str(n) + ":"
            else:
                if self.system_prompt in ["style_and_length_v2"]:
                    prefix = style + ":"
                else:
                    prefix = style + " :"
        elif self.system_prompt == "no_style":
            prefix = ""
        else:
            raise NotImplementedError(self.system_prompt)

        return prefix

    def format_input_timestamps(self, rng, timestamps):
        """Format input timestamp as text"""
        timestamps = [parse_timestamp(x) for x in timestamps]
        if self.timestamp_mode == "rng-v1":
            raise NotImplementedError()
        if self.timestamp_mode == "50-percent-seconds":
            if rng.random() > 0.5:
                return True, [str(round(x, self.seconds_decimal_places)) for x in timestamps]
            else:
                return False, [seconds_to_timestamp(x, self.seconds_decimal_places) for x in timestamps]
        elif self.timestamp_mode == "seconds-to-tenth":
            return True, [str(round(x, 1)) for x in timestamps]
        elif self.timestamp_mode == "seconds":
            return True, [str(round(x, self.seconds_decimal_places)) for x in timestamps]
        else:
            raise NotImplementedError()

    def format_output_timestamp(self, time_value):
        """Format output timestamp as text

        For output timestamps, the model should always use a consistent format
        """
        time_value = parse_timestamp(time_value)
        if isinstance(time_value, str):
            return time_value
        else:
            if self.output_timestamp_mode == "timestamp":
                return seconds_to_timestamp(time_value, self.seconds_decimal_places)
            elif self.output_timestamp_mode == "seconds":
                return str(round(time_value, self.seconds_decimal_places))
            else:
                raise NotImplementedError()

    def get_user_prompt(self, example, is_training=True, for_inference=False, rng=None):
        """Build a list of strings of what a user might type in to the model for the given example,
        and its responses, by applying a prompt template to the fields in `example`

        Uses the `style` field to understand what the task/output style is
        """
        video_object_track_styles = [
            "video_point_track_per_frame", 
            "video_point_ground_start_end", 
            "video_single_point_track_per_frame",
        ]
        multi_image_pointing_styles = [
            "multi_image_pointing",
            "multi_image_counting",
            "multi_image_point_then_count",
            "multi_image_count_then_point",
        ]

        style = example.get("style")
        output = None
        metadata = None
        if "prompt" in example:
            # Examples have a complete user prompt pre-specified, usually for eval sets
            prompt = example["prompt"]

        elif self.prompt_templates == "none":
            # Bare-bone prompt with no templating or instructions
            if "prompt" in example:
                prompt = example["prompt"]
            elif style in ["pointing", "point_count", "point_then_count", "cosyn_point"]:
                if "question" in example:
                    prompt = example["question"]
                else:
                    if "label" in example:
                        prompt = example["label"]
                        prompt = prompt.lower()
                    else:
                        prompt = example["label_cased"]
                output = self.format_points(example)
            elif "question" in example and ("options" in example or "unlabelled_options" in example):
                output, prompt, metadata = self.format_options(example)
            elif "timestamp" in example:
                prompt = str(round(example["timestamp"], 2))
            elif "start_time" in example:
                prompt = str(round(example["start_time"], 2)) + "-" + str(round(example["end_time"], 2))
            elif "question" in example:
                prompt = example["question"]
            else:
                prompt = ""

        elif self.prompt_templates in ["uber_model", "uber_model_v2"]:
            if self.prompt_templates == "uber_model_v2":
                template_all_multiple_choice = True
            else:
                template_all_multiple_choice = False

            # We template long captions and pointing since they are "demo" tasks, and use
            # plain text for everything else
            if style in [
                "long_caption",
                "short_caption",
                "video_long_caption",
                "video_short_caption",
                "video_transcript",
                "video_motion_caption",
                "video_object_caption",
            ] and "question" not in example:
                prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1[style], example, rng, dbg=self.debug)
            elif style == "video_frame_caption_timestamp":
                in_seconds, (timestamp_str,) = self.format_input_timestamps(rng, [example["timestamp"]])
                prompt = apply_keyword_prompt(
                    GENERAL_PROMPTS_V1[style + ("_in_seconds" if in_seconds else "")],
                    dict(example, timestamp=timestamp_str),
                    rng, dbg=self.debug)
            elif style in ["video_clip_caption_start_end", "video_clip_transcript_start_end"]:
                in_seconds, (start_str, end_str) = self.format_input_timestamps(
                    rng, [example["start_time"], example["end_time"]])
                prompt = apply_keyword_prompt(
                    GENERAL_PROMPTS_V1[style + ("_in_seconds" if in_seconds else "")],
                    dict(example, start_time=start_str, end_time=end_str),
                    rng, dbg=self.debug)
            elif style in ["video_short_answer", "video_short_answer_count_without_pointing"]:
                prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1[style], example, rng, dbg=self.debug)
            elif "_exp" in style:
                prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1["chain_of_thought"], example, rng, dbg=self.debug)
            elif style in ["pointing", "point_count", "point_then_count", "cosyn_point"]:
                if "question" in example:
                    prompt = example["question"]
                else:
                    if "label" in example:
                        prompt = example["label"].lower()
                    else:
                        prompt = example["label_cased"]
                    prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1[style], dict(example, label=prompt), rng, dbg=self.debug)
                output = self.format_points(example)
            elif style in video_object_track_styles:
                prompt, output = self.format_video_object_track_points(example, is_training, rng)
            elif style in [
                    "video_point",
                    "video_point_count",
                    "video_count",
                    "video_count_point",
                ]:
                if "question" in example:
                    prompt = example["question"]
                else:
                    prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1[style], example, rng, dbg=self.debug)
                output = self.format_video_points(example)
                metadata = example.get("metadata", {})
                metadata["answer"] = output
            elif style in multi_image_pointing_styles:
                prompt, output = self.format_multi_points(example)
                metadata = example.get("metadata", {})
                metadata["answer"] = output
            elif "prompt" in example:
                prompt = example["prompt"]
            elif "question" in example and ("options" in example or "unlabelled_options" in example):
                if template_all_multiple_choice:
                    prompt, output, metadata = self.template_options(example, is_training, rng)
                else:
                    output, prompt, metadata = self.format_options(example)
                    if style in VIDEO_MC_STYLES and self.template_video_mc_questions:
                        # apply template on top of formatted options
                        options_text = prompt.replace(example["question"], "").strip()
                        prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1[style], dict(example, question=example["question"], options=options_text), rng, dbg=self.debug)
            elif "question" in example:
                prompt = example["question"]
            else:
                prompt = ""
        else:
            raise NotImplementedError(self.prompt_templates)

        if output is None and not for_inference:
            if "answers" in example:
                output = self.select_vqa_answer(example["answers"], rng)
            elif "answer" in example:
                output = example["answer"]
                if "answer_annotations" in example:
                    output = self.format_annotated_text(output, example["answer_annotations"])
                elif "explanation" in example:
                    output = example["explanation"] + " Answer: " + output
            elif "answer_with_points" in example:
                output = example["answer_with_points"]
            elif "text" in example:
                output = example["text"]
            else:
                print(example)
                raise ValueError("No output in example, if this is an inference-only task make sure `for_inference` is True")
        return prompt, output, metadata

    def _format_example(self, message, example, is_training, for_inference, rng):
        metadata = {}
        for k in ["answer_idx", "answers", "answer", "points", "options"]:
            if k in message:
                metadata[k] = message[k]
        if isinstance(message, str):
            messages = [message]
        elif isinstance(message, list):
            messages = message
        elif "messages" in message:
            # Example directly contains the prompts/message to use
            messages = message["messages"]
        elif isinstance(message, dict):
            # An example that requires a custom prompt
            if "video" in example:
                video = example["video"]
                if hasattr(video, "timestamps"):  # JAMES: use loaded video to sample point tracks with aligned timestamps and fps
                    message["video"] = {"timestamps": video.timestamps, "target_fps": video.target_fps}
            if "multi_turn_messages" in example:
                messages = []
                # multi-turn conversations that needs to be formatted through `get_user_prompt`
                for turn_message in message["multi_turn_messages"]:
                    prompt, response, extra_metadata = self.get_user_prompt(
                        turn_message, is_training, for_inference=for_inference, rng=rng
                    )
                    assert response is not None
                    messages += [prompt, response]
            else:
                prompt, response, extra_metadata = self.get_user_prompt(
                    message, is_training, for_inference=for_inference, rng=rng
                )
                if extra_metadata:
                    metadata.update(extra_metadata)
                if not for_inference:
                    assert response is not None
                    messages = [prompt, response]
                else:
                    messages = [prompt]
        else:
            raise ValueError(f"Example type {type(message)} not understood")

        # Add the system prompt
        if self.system_prompt and self.system_prompt != "none":
            style = None
            if isinstance(message, dict):
                if "multi_turn_messages" in message:
                    # FIXME This is a bit of hack, its okay for now since our only multi-turn
                    # messages use the "demo" style
                    style = message["multi_turn_messages"][0]["style"]
                else:
                    style = message.get("style", None)
            prefix = self.get_system_prompt(style, for_inference, messages, is_training, rng=rng)
            if len(prefix) > 0 and len(messages[0]) > 0:
                with_system_prompt = prefix + " " + messages[0]
            elif len(prefix) > 0:
                with_system_prompt = prefix
            else:
                with_system_prompt = messages[0]
            messages = [with_system_prompt] + messages[1:]

        if (
            self.image_last and
            ("image" in example or "video" in example) and
            tokenizer.IMAGE_PROMPT not in messages[0]
        ):
            messages[0] = messages[0] + tokenizer.IMAGE_PROMPT

        # Add the role annotations such as "User:" and "Assistant:"
        messages = self.format_messages(messages)
        return messages, metadata

    def __call__(self, ex: Dict, is_training, for_inference, rng) -> Tuple[Dict, Dict]:
        """Returns a formatted example and example metadata"""
        if "message_list" in ex:
            if self.p_one_message and rng.random() < self.p_one_message:
                ex["message_list"] = ex["message_list"][:1]
            elif self.format_message_list == "numbered_qa":
                ex["message_list"] = [dict(x) for x in ex["message_list"]]
                for ix, msg_list in enumerate(ex["message_list"], start=1):
                    msg_list["question"] = f"{' ' if ix != 0 else ''}Q{ix}: {msg_list['question']}"
                    msg_list["answer"] = f"A{ix}: " + msg_list["answer"]
            else:
                assert self.format_message_list is None
        
        if "message_list" in ex:
            # Does not support returning metadata, which is fine since we are not doing inference
            return [self._format_example(msg, ex, is_training, for_inference, rng)[0]
                    for msg in ex["message_list"]], None
        elif "messages" in ex:
            return self._format_example(ex["messages"], ex, is_training, for_inference, rng)
        else:
            return self._format_example(ex, ex, is_training, for_inference, rng)
