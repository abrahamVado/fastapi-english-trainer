import React from "react";
export default function App() {
  const rawHtml = \`
    <div style="padding:1rem;border:1px solid #ccc;border-radius:8px;">
      <h1>English Trainer Playground</h1>
      <p>This is your placeholder. Paste your HTML snippet here.</p>
    </div>
  \`;
  return <div dangerouslySetInnerHTML={{ __html: rawHtml }} />;
}
