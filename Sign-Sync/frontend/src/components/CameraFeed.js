import React from "react";

export default function CameraFeed({ videoRef, width = 640, height = 400, className = "" }) {
  return (
    <div className={className}>
      <video ref={videoRef} autoPlay playsInline width={width} height={height} />
    </div>
  );
}
