import React, { useCallback, useRef, useState } from "react";

const ACCEPTED = ["application/pdf", "image/jpeg", "image/png"];
const ACCEPTED_EXT = [".pdf", ".jpg", ".jpeg", ".png"];
const MAX_MB = 50;

export default function DropZone({ onFile }) {
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef();

  const validate = (file) => {
    if (!ACCEPTED.includes(file.type)) {
      setError(`Unsupported type: ${file.type || file.name}`);
      return false;
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      setError(`File exceeds ${MAX_MB} MB`);
      return false;
    }
    setError(null);
    return true;
  };

  const handle = useCallback(
    (file) => {
      if (validate(file)) onFile(file);
    },
    [onFile]
  );

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handle(file);
  };

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current.click()}
      className={`cursor-pointer rounded-xl border-2 border-dashed p-10 text-center transition-colors
        ${dragging ? "border-blue-400 bg-blue-950/30" : "border-gray-600 hover:border-gray-400"}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_EXT.join(",")}
        className="hidden"
        onChange={(e) => { if (e.target.files[0]) handle(e.target.files[0]); }}
      />
      <p className="text-lg text-gray-300">Drop a PDF or image here</p>
      <p className="mt-1 text-sm text-gray-500">PDF, JPG, PNG — max {MAX_MB} MB</p>
      {error && <p className="mt-3 text-sm text-red-400">{error}</p>}
    </div>
  );
}
