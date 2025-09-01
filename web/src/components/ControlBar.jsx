// src/components/ControlBar.jsx
import React from "react";
import RecordButton from "./RecordButton.jsx";

export default function ControlBar({
  recording,
  processing,
  onMicClick,
  onStart,
  onNext,
  onScore,
  canNext,
  canScore,
}) {
  return (
    <div className="controlbar">
      {/* Mic */}
      <div className="tooltip">
        <RecordButton recording={recording} onClick={onMicClick} />
        <span className="tooltip-text">
          {recording ? "Stop & Send" : "Start Recording"}
        </span>
      </div>

      {/* Start */}
      <div className="tooltip">
        <button
          className="sim-btn sim-btn--start"
          onClick={onStart}
          disabled={processing}
          aria-label="Start session"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path fill="white" d="M2 21l21-9L2 3v7l15 2-15 2v7z" />
          </svg>
        </button>
        <span className="tooltip-text">Start a new session</span>
      </div>

      {/* Next */}
      <div className="tooltip">
        <button
          className="sim-btn sim-btn--next"
          onClick={onNext}
          disabled={!canNext || processing}
          aria-label="Next question"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path fill="white" d="M10 17l5-5-5-5v10zM4 5h2v14H4V5zm14 0h2v14h-2V5z" />
          </svg>
        </button>
        <span className="tooltip-text">Ask next question</span>
      </div>

      {/* Score */}
      <div className="tooltip">
        <button
          className="sim-btn sim-btn--score"
          onClick={onScore}
          disabled={!canScore || processing}
          aria-label="Score answer"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path fill="white" d="M9 16.17l-3.88-3.88L4 14.41 9 19.41 20 8.41 18.59 7l-9.59 9.59z" />
          </svg>
        </button>
        <span className="tooltip-text">Score your answer</span>
      </div>
    </div>
  );
}
