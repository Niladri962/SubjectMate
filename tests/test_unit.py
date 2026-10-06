"""Fast tests that need no index, model download or API key."""
import pytest
from langchain_core.documents import Document

from subjectmate import llm
from subjectmate.rag import (format_history, is_follow_up, remove_unsupported_citations,
                             standalone_question)
from subjectmate.retriever import BM25, tokenize


def doc(source: str, page: int) -> Document:
    return Document(page_content="text", metadata={"source": source, "page": page})


class TestCitations:
    def test_keeps_citation_of_retrieved_page(self):
        text = "PCA reduces dimensions [LA_Lecture 10_SM.pdf, p. 3]."
        assert remove_unsupported_citations(text, [doc("LA_Lecture 10_SM.pdf", 3)]) == text

    def test_drops_invented_source(self):
        text = "It projects data. [LA_Lecture 11.pdf, p. 5]"
        assert remove_unsupported_citations(text, [doc("LA_Lecture 10_SM.pdf", 3)]) == "It projects data."

    def test_drops_wrong_page_of_retrieved_file(self):
        text = "Claim [L19.pdf, p. 99]."
        assert remove_unsupported_citations(text, [doc("L19.pdf", 3)]) == "Claim."

    def test_keeps_slide_citations(self):
        text = "A p-value is ... [Hypothesis Test.pptx, slide 9]"
        assert remove_unsupported_citations(text, [doc("Hypothesis Test.pptx", 9)]) == text


class TestKeywordSearch:
    def test_tokenize_folds_plurals_and_drops_stopwords(self):
        assert tokenize("The queues and trees") == ["queue", "tree"]
        assert "is" not in tokenize("what is a stack")

    def test_bm25_ranks_matching_document_first(self):
        bm25 = BM25(["stack push pop", "queue enqueue dequeue", "binary tree traversal"])
        scores = bm25.scores("how does a queue work")
        assert scores.argmax() == 1
        assert scores[0] == 0


class TestFollowUps:
    history = [{"role": "user", "content": "How does Dijkstra's algorithm work?"},
               {"role": "assistant", "content": "It finds shortest paths..."}]

    @pytest.mark.parametrize("question", ["What is its time complexity?", "why?", "and Floyd-Warshall?",
                                          "Give an example"])
    def test_detects_follow_ups(self, question):
        assert is_follow_up(question, self.history)

    def test_standalone_question_is_not_a_follow_up(self):
        assert not is_follow_up("How does a circular queue reuse empty space in an array?", self.history)

    def test_no_history_means_no_follow_up(self):
        assert not is_follow_up("why?", [])

    def test_offline_rewrite_adds_previous_question(self):
        query = standalone_question("What is its time complexity?", self.history, "extractive", None, None)
        assert "Dijkstra" in query and "time complexity" in query

    def test_history_is_truncated_to_recent_turns(self):
        long_history = [{"role": "user", "content": f"q{i}"} for i in range(20)]
        text = format_history(long_history)
        assert "q19" in text and "q0" not in text
        assert format_history([]) == ""


class TestProviders:
    def test_missing_key_is_a_clear_error(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(llm.LLMError, match="No API key"):
            llm.generate("system", "user", provider="groq")

    def test_unknown_provider(self):
        with pytest.raises(llm.LLMError, match="Unknown provider"):
            llm.generate("system", "user", provider="nope")

    def test_extractive_needs_no_key(self):
        assert llm.PROVIDERS["extractive"].ready()
