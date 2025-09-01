export default function RecordButton({ recording, onClick }) {
  return (
    <button
      id="recordButton"
      className="record-btn"
      aria-label={recording ? "Stop and send" : "Start recording"}
      aria-pressed={recording ? "true" : "false"}
      onClick={onClick}
    >
      {recording ? (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path fill="white" d="M6 6h12v12H6z" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path
            fill="white"
            d="M12 14a3 3 0 0 0 3-3V6a3 3 0 1 0-6 0v5a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 14 0h-2zm-5 9a7 7 0 0 1-7-7H3a9 9 0 0 0 18 0h-2a7 7 0 0 1-7 7z"
          />
        </svg>
      )}
    </button>
  );
}
