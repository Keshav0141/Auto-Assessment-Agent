# Custom test case: unseen paper (machine learning + general knowledge)

A question paper that is not one of the built-in samples, used to show that the
grader works on any paper, subject and rubric.

## How to run it

Upload each file into its sheet (or paste its text), then press **Mark the script**:
question_paper.txt -> Question paper, answer_sheet.txt -> Answer sheet,
rubric.txt -> Rubric, model_answers.txt -> Model answers.

## Expected result

- Q1: about 3/4. Correct definition and one method (dropout); the second method is missing.
- Q2: about 2/3. Names Rabindranath Tagore; the year (1911) is missing.

The feedback should name the missing criterion in each question.

## Format notes

- Each question starts with a "Question N" label; marks go in brackets on the label line.
- In the rubric, criteria are separated by commas, so a criterion must not contain a comma.
