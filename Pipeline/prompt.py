def prompt_qac_wiki(context):
    context = '\n'.join(f'{i}: {c}' for i, c in enumerate(context, start=1))

    start = "Given the following question and contexts, create a final answer to the question."

    return start + '\n=========\n' + 'QUESTION: {question}' + '\n=========\n' + 'CONTEXT:\n' + context + '\n=========\n' + 'QUESTION: {question}' + '\n=========\n' + 'ANSWER: please answer less than 6 words.'


def prompt_qa_wiki(context):
    start = "Given the following question, create a final answer to the question."

    return start + '\n=========\n' + 'QUESTION: {question}' + '\n=========\n' + 'ANSWER: please answer less than 6 words.'


def prompt_eval():
    eval_prompt = """You are an expert professor specialized in grading whether the prediction to the question is correct or not according to the real answer.
    ==================
    For example:
    ==================
    Question: What company owns the property of Marvel Comics?
    Answer: The Walt Disney Company
    Prediction: The Walt Disney Company
    Return: 1
    ==================
    Question: Which constituent college of the University of Oxford endows four professorial fellowships for sciences including chemistry and pure mathematics?
    Answer: Magdalen College
    Prediction: Magdalen College.
    Return: 1
    ==================
    Question: Which year was Marvel started?
    Answer: 1939
    Prediction: 1200
    Return: 0
    ==================
    You are grading the following question:
    Question: {question}
    Answer: {answer}
    Prediction: {prediction}
    If the prediction is correct according to answer, return 1. Otherwise, return 0.
    Return: your reply can only be one number '0' or '1'
    """

    return eval_prompt



if __name__ == '__main__':
    context = ['I am a student at Vanderbilt', 'My name is Yu']
    question = 'What is my name?'

    print(prompt_qac_wiki(context, question))
