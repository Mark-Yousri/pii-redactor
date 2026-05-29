import React, { useState } from "react";

function PageNav({ page, total, onPrev, onNext }) {
  if (total <= 1) return null;
  return (
    <div className="flex items-center gap-3 text-sm text-gray-400 mt-2">
      <button onClick={onPrev} disabled={page === 0} className="disabled:opacity-30 hover:text-white">‹ Prev</button>
      <span>{page + 1} / {total}</span>
      <button onClick={onNext} disabled={page === total - 1} className="disabled:opacity-30 hover:text-white">Next ›</button>
    </div>
  );
}

function Skeleton() {
  return (
    <div className="animate-pulse bg-gray-800 rounded-lg w-full aspect-[3/4]" />
  );
}

export default function PreviewPane({ originalPages, redactedPages, loading }) {
  const [page, setPage] = useState(0);
  const total = Math.max(originalPages?.length ?? 0, redactedPages?.length ?? 0, 1);

  return (
    <div className="grid grid-cols-2 gap-4">
      {/* Original */}
      <div>
        <p className="mb-2 text-xs uppercase tracking-widest text-gray-500">Original</p>
        {originalPages && originalPages[page] ? (
          <img
            src={`data:image/png;base64,${originalPages[page]}`}
            alt={`Original page ${page + 1}`}
            className="w-full rounded-lg border border-gray-700"
          />
        ) : (
          <div className="w-full aspect-[3/4] rounded-lg border border-gray-700 bg-gray-900 flex items-center justify-center text-gray-600 text-sm">
            Upload a file
          </div>
        )}
      </div>

      {/* Redacted */}
      <div>
        <p className="mb-2 text-xs uppercase tracking-widest text-gray-500">Redacted</p>
        {loading ? (
          <Skeleton />
        ) : redactedPages && redactedPages[page] ? (
          <img
            src={`data:image/png;base64,${redactedPages[page]}`}
            alt={`Redacted page ${page + 1}`}
            className="w-full rounded-lg border border-gray-700"
          />
        ) : (
          <div className="w-full aspect-[3/4] rounded-lg border border-dashed border-gray-700 bg-gray-900 flex items-center justify-center text-gray-600 text-sm">
            Result appears here
          </div>
        )}
      </div>

      <div className="col-span-2 flex justify-center">
        <PageNav
          page={page}
          total={total}
          onPrev={() => setPage((p) => Math.max(0, p - 1))}
          onNext={() => setPage((p) => Math.min(total - 1, p + 1))}
        />
      </div>
    </div>
  );
}
