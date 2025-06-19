# utils/helpers.py
def is_similar_question(new_question, existing_questions, similarity_threshold=0.7):
    if not existing_questions:
        return False
    new_words = set(new_question.lower().split())
    for existing in existing_questions:
        existing_words = set(existing.lower().split())
        intersection = len(new_words & existing_words)
        union = len(new_words | existing_words)
        if union > 0 and intersection / union > similarity_threshold:
            return True
    return False
