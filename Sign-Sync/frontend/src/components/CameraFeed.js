import React from "react";

export default function CameraFeed({ videoRef, className = "" }) {
  return (
    <div className={`relative ${className}`}>
      <div className="aspect-[16/9] bg-black/30 rounded-2xl border border-black/10 dark:border-white/10 overflow-hidden">
        <video ref={videoRef} autoPlay playsInline className="w-full h-full object-cover" />
      </div>
    </div>
  );
}
