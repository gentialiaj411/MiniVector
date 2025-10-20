import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [latency, setLatency] = useState(0);
  const [selectedArticle, setSelectedArticle] = useState(null);
  const [articleLoading, setArticleLoading] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    try {
      const response = await axios.post("http://localhost:8000/search", {
        query: query,
        k: 10,
      });
      setResults(response.data.results);
      setLatency(response.data.took_ms);
    } catch (error) {
      console.error("Search failed:", error);
    }
    setLoading(false);
  };

  const viewArticle = async (articleId) => {
    try {
      const response = await axios.get(
        `http://localhost:8000/article/${articleId}`
      );
      setSelectedArticle(response.data);
    } catch (error) {
      console.error("Failed to load article:", error);
    }
  };

  return (
    <div className="App">
      {selectedArticle ? (
        <div className="article-modal">
          <div className="modal-content">
            <button
              onClick={() => setSelectedArticle(null)}
              className="close-button"
            >
              ← Back to Search
            </button>
            <div className="article-full">
              <span
                className={`category category-${selectedArticle.category.toLowerCase()}`}
              >
                {selectedArticle.category}
              </span>
              <h1>{selectedArticle.title}</h1>
              <div className="article-text">
                {articleLoading
                  ? "Loading article…"
                  : selectedArticle.text ||
                    selectedArticle.full_text ||
                    selectedArticle.text_preview ||
                    "No text available."}
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="container">
          <h1>🔍 MiniVector Search</h1>
          <p className="subtitle">Semantic search over 100,000 news articles</p>

          <div className="search-box">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && handleSearch()}
              placeholder="Search for anything..."
              className="search-input"
            />
            <button
              onClick={handleSearch}
              disabled={loading}
              className="search-button"
            >
              {loading ? "Searching..." : "Search"}
            </button>
          </div>

          {latency > 0 && (
            <p className="latency">
              Found {results.length} results in {latency.toFixed(1)}ms
            </p>
          )}

          <div className="results">
            {results.map((result, index) => (
              <div key={result.id} className="result-card">
                <div className="result-header">
                  <span className="result-number">#{index + 1}</span>
                  <span
                    className={`category category-${result.category.toLowerCase()}`}
                  >
                    {result.category}
                  </span>
                  <span className="score">
                    Score: {result.score.toFixed(3)}
                  </span>
                </div>
                <h3 className="result-title">{result.title}</h3>
                <p className="result-text">{result.text_preview}</p>
                <button
                  onClick={() => viewArticle(result.id)}
                  className="view-article-button"
                >
                  View Full Article →
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
