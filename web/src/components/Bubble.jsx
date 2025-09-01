export default function Bubble({ m }) {
  const isUser = m.role === "user";
  return (
    <div className={`message ${m.role}`}>
      <div className={`avatar ${m.role}`}>
        {isUser ? (
          <svg viewBox="0 0 24 24">
            <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm0-12.5c-1.38 0-2.5 1.12-2.5 2.5s1.12 2.5 2.5 2.5 2.5-1.12 2.5-2.5-1.12-2.5-2.5-2.5zm0 7.5c-2.21 0-4 1.79-4 4h8c0-2.21-1.79-4-4-4z" />
          </svg>
        )}
      </div>

      <div className={`bubble ${m.role}`}>
        <div className="footer">
          <span>
            {new Date(m.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </span>
        </div>
        <div className="divider" />
        <div className="content">
          {m.type === "text" && m.payload}
          {m.type === "audio" && <audio controls src={URL.createObjectURL(m.payload)} />}
        </div>
      </div>
    </div>
  );
}
