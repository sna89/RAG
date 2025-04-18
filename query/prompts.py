"""
Prompts for various NLP tasks in the QA system.

This module contains prompt templates organized by functionality:
- RAG (Retrieval-Augmented Generation)
- Topic Classification
- Query Processing
- Summarization
- Evaluation
"""

# ==============================
# RAG Prompts
# ==============================

RAG_PROMPT = """
    RAG_PROMPT = You are an assistant for question-answering tasks. Use the following pieces of retrieved context to 
    answer the question. If you don't know the answer, just say that you don't know. 
    
    If the question is composed of several questions, follow this step-by-step reasoning process:
    1. Identify each distinct question within the user's query
    2. For each identified question:
       a. State the specific question being addressed
       b. Analyze what information is needed to answer it
       c. Examine the relevant retrieved documents for this specific question
       d. Reason through the answer by connecting the information to the question
       e. Formulate a clear conclusion for this specific question
    3. After addressing each question individually, synthesize the separate answers into a cohesive summary that 
    addresses the overall query
    4. Ensure the final response maintains logical connections between the individual questions
    
    Use maximum three sentences, but try to answer in exactly one sentence, and keep the answer short and concise.
    In addition, don't add information that is not relevant to the current question.
    
    Question: {question} 
    Context: {context} 
    Answer:
    """

MULTIQUERY_PROMPT = """
You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}
"""

# ==============================
# Classification Prompts
# ==============================

TOPIC_CLASSIFICATION_PROMPT = """You are an expert classifier with access to domain knowledge in the 
    following four fields:
    Automobile, Food, Steel, Textile.
    Your task is to classify each input query into one or several of the following five categories:
    Automobile, Food, Steel, Textile or General

    Use the following rules:

    1. If the query clearly relates to one of the four fields, assign that field.
    2. If the query can be related to several from the four mentioned fields, assign it to all relevant fields. 
    3. If the query is ambiguous, even if it seems to relate to a known field (e.g. vague or unclear intent), 
    classify it as Ambiguous.
    4. If the query is not ambiguous and it is not specific to one of the fields, classify it as General.
    5. Give most importance to the query itself, then to the last part of the conversation.
    Consider previous content as well.

    Return only the name of the selected categories.

    Example 1
    Query: "What are the emission standards for diesel engines?"
    Category: Automobile
       
    Example 2
    Query: "Which main transformations occurred in the 19th century?."
    Category: General
    
    Example 3:
    Query: "which actions initiated the transformations in the food and textile industries?"
    Category: Food, Textile
    
    Query: {query}
    Category:
    """

# ==============================
# Ambiguity Detection Prompts
# ==============================

AMBIGUOUS_PROMPT_1 = """
    Given a possible context and a query, classify the question as either:
    General: Broad but understandable and answerable, with or without the current context.
    Ambiguous: Vague, unclear, or needs more context to be answered reliably.
    
    Example 1
    Conversation Context:
    A: I'm planning a trip to Japan this summer.
    B: That sounds exciting! Where are you going exactly?

    Follow-up Question:
    What should I pack?
    
    Classification:
    General

    Example 2
    Conversation Context:
    A: I’m thinking of switching careers.
    B: Oh? What kind of field are you considering?
    
    Follow-up Question:
    What do you think?
    
    Classification:
    Ambiguous
    
    Conversation Context:
    {context}

    Follow-up Query:
    {query}

    Output only the classification:
    True if Ambiguous else False"""

AMBIGUOUS_PROMPT = """
You are tasked with assessing whether a given query is ambiguous or not.

Criteria for Ambiguity:
A query is considered ambiguous if it meets any of the following conditions:

Lack of Context: Essential information needed to understand or answer the query is missing.

Vagueness: The query includes generic or undefined terms without additional clarification.

Unclear Scope: The query lacks a clear subject, timeframe, or domain.

Contradictions: It contains conflicting or self-contradictory statements.

Multiple Interpretations: The query could apply to multiple unrelated topics or meanings.

If the query is specific, well-defined, and self-contained — with enough information to generate a meaningful answer — it is not ambiguous.

Output Instructions:
Provide a confidence score between 0.0 and 1.0, where:

0.0 = clearly not ambiguous

1.0 = clearly ambiguous

Values in between indicate partial ambiguity

Respond with the score only, without explanations.

Query:  
{query}

Score:
"""

# ==============================
# Query Processing Prompts
# ============================

SPLIT_PROMPT = """
    You are an assistant that rewrites compound or multi-part queries into separate, clearly phrased questions, 
    ensuring that each sub-question preserves full context from the original query.
    
    Guidelines:
    1. Split the input into distinct questions based on the logical structure.
    2. Each sub-question must be grammatically correct and standalone, meaning it should include necessary context 
    from the original query.
    3. Maintain the original meaning and intent.
    4. Please return the result as a list.
    
    Example
        Original Query: "When did the food transformation started and how it is began?"
    Split Questions:    
        1. When did the food transformation start?
        2. How did the food transformation began?
    
    Original Query: "{query}"
    Split Questions:
    """

GRAMMAR_FIX_PROMPT = """
    You are a professional copy editor.
    Your task is to correct grammar, spelling, punctuation, and phrasing errors in the given text. 
    Maintain the original meaning and tone. 
    Do not rewrite the text unless necessary for correctness or clarity.
    Return only the corrected version.

    Text: 
    {query}

    Output: 
"""


# ==============================
# Summarization Prompts
# ==============================

SUMMARIZE_CONVERSATION_PROMPT = """
    You are a professional summarizer specialized in summarizing QA-style conversations between a human and 
    a language model.
    Your task is to read the following dialogue and generate a concise, paragraph-based summary that captures the core 
    questions asked, and the model’s responses.
    Focus on the flow of the exchange, highlight the intent behind the user’s questions, 
    and the completeness or relevance of the model's answers.
    Do not use bullet points or numbered lists. Write only clean, coherent paragraphs.
    Keep the tone formal, neutral, and informative.
    
    Here is the conversation:
    
    Conversation:
    {conversation}  
    
    Provide the summary below:"""


SUMMARIZE_DOCUMENT_PROMPT = """
    You are a professional summarizer. Your goal is to read a document and write a concise and coherent summary 
    that captures the key points and main ideas.
    
    Guidelines:
    - Do not copy entire sentences from the document.
    - Focus on the most important information, ideas, and arguments.
    - Omit minor details, examples, or tangents.
    - Preserve the meaning and intent of the original text.
    - Use clear and neutral language.
    - When need, explain and elaborate about the core main ideas
    
    Document:
    {document_text}
    
    Summary:
"""

# SUMMARY_QUERY_COMBINE_PROMPT = """
#     You are a professional assistant that takes a conversation summary and a new user query, and combines them into
#     a single, coherent, and context-rich instruction, that will be sent to an LLM.
#
#     Your task is to generate an optimized search query that will retrieve the most relevant
#     documents from the knowledge base.
#
#     CONVERSATION HISTORY:
#     {conversation_history}
#
#     NEW USER QUERY:
#     {query}
#
#     Instructions:
#     1. Extract key topics, entities, and information needs from both the conversation history and new query
#     2. Identify how the new query relates to or builds upon previous exchanges
#     3. Determine if the new query introduces new topics or is requesting clarification/elaboration on previously
#     discussed topics
#     4. Synthesize a comprehensive search query that captures the full context of the user's information need
#     5. Include essential keywords, technical terms, and any specific constraints mentioned in the conversation
#
#     OUTPUT:
#     Generate an optimized search query that will retrieve the most relevant documents to answer the user's
#     current question in context.
#     """

SUMMARY_QUERY_COMBINE_PROMPT = """
    Given a chat history and the latest user question which might reference context in the chat history, 
    formulate a standalone question which can be understood without the chat history. 
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
        
    chat_summary_history:
    {conversation_history}
    
    query:
    {query}
"""


# ==============================
# Evaluation Prompts
# ==============================

LLM_AS_JUDGE_ACC_PROMPT = """
    You are a fair and strict evaluator. 
    Given a generated answer, and the ground truth answer, return a similarity score between 0 and 1.

    The score should reflect how closely the generated answer matches the ground truth in terms of meaning.
    
    - Return 1 if they are nearly identical or semantically equivalent.
    - Return 0 if they are completely different or irrelevant.
    - Use intermediate values to reflect partial correctness or partial relevance.
    - Only return a number between 0 and 1, with no explanation.
    
    Examples:
    
    Generated Answer: Climate change reduces sea ice, which limits polar bears' ability to hunt and find food.
    Ground Truth: Due to melting sea ice caused by climate change, polar bears struggle to hunt seals and maintain their nutrition.
    Score: 1
    
    Generated Answer: Polar bears are cute animals that live in the Arctic.
    Ground Truth: Due to melting sea ice caused by climate change, polar bears struggle to hunt seals and maintain their nutrition.
    Score: 0
    
    Generated Answer: The industrial revolution led to rapid urban growth as people moved to cities for factory jobs.
    Ground Truth: The rise of factories during the industrial revolution attracted rural populations to urban centers, accelerating urbanization.
    Score: 0.95
    
    Generated Answer: It created many machines.
    Ground Truth: The rise of factories during the industrial revolution attracted rural populations to urban centers, accelerating urbanization.
    Score: 0.2
    
    Now evaluate:
    
    Generated Answer: {generated_answer}
    Ground Truth: {ground_truth}
    Score:"""

LLM_AS_JUDGE_REL_PROMPT = """
    You are an evaluator assessing how relevant an answer is to a given question. 
    Return a score between 0 and 1.
    
    - Return 1 if the answer directly and completely addresses the question.
    - Return 0 if the answer is off-topic or completely unrelated.
    - Use values in between for partially relevant or vague answers.
    - Do not provide any explanation. Only return a number between 0 and 1.
    
    Examples:
    
    Question: What causes rainfall?
    Answer: Rainfall is caused by the condensation of water vapor in the atmosphere.
    Score: 1
    
    Question: What causes rainfall?
    Answer: Rainfall is important for crops.
    Score: 0.3
    
    Question: What causes rainfall?
    Answer: I enjoy rainy days.
    Score: 0
    
    Now evaluate:
    
    Question: {question}
    Answer: {answer}
    Score:"""


