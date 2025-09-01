export default function StatusChip({ kind = "", text = "" }) {
  return (
    <div className="status-chip" aria-live="polite">
      <span className={`dot${kind ? ` ${kind}` : ""}`} />
      <span>{text}</span>
    </div>
  );
}
