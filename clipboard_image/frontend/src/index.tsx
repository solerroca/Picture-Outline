// @ts-nocheck
import React, { useEffect, useRef, useState } from "react";
import ReactDOM from "react-dom";
import { Streamlit, withStreamlitConnection } from "streamlit-component-lib";

const ClipboardImage: React.FC = () => {
  const zoneRef = useRef<HTMLDivElement | null>(null);
  const [message, setMessage] = useState<string>(
    "Click here then paste (Ctrl+V / Cmd+V)"
  );
  const [bg, setBg] = useState<string>("#f8fff8");
  const [border, setBorder] = useState<string>("#4CAF50");

  useEffect(() => {
    const zone = zoneRef.current;
    if (!zone) return;

    const handleClick = () => zone.focus();
    const handlePaste = (e: ClipboardEvent) => {
      e.preventDefault();
      const items = e.clipboardData?.items || [];
      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.type.indexOf("image") !== -1) {
          const blob = item.getAsFile();
          if (blob) {
            const reader = new FileReader();
            setBg("#fff3cd");
            setBorder("#ffc107");
            setMessage("Processing image…");
            reader.onload = evt => {
              const b64 = evt.target?.result as string;
              Streamlit.setComponentValue({ data: b64, ts: Date.now() });
              setBg("#d4edda");
              setBorder("#28a745");
              setMessage("Screenshot pasted! You can paste another one.");
            };
            reader.readAsDataURL(blob);
          }
          return;
        }
      }
      setBg("#f8d7da");
      setBorder("#dc3545");
      setMessage("No image in clipboard – copy an image first");
    };

    zone.addEventListener("click", handleClick);
    zone.addEventListener("paste", handlePaste as any);
    return () => {
      zone.removeEventListener("click", handleClick);
      zone.removeEventListener("paste", handlePaste as any);
    };
  }, []);

  return (
    <div
      ref={zoneRef}
      tabIndex={0}
      style={{
        border: `2px dashed ${border}`,
        padding: 20,
        textAlign: "center",
        borderRadius: 8,
        background: bg,
        fontFamily: "Arial, sans-serif",
        userSelect: "none",
        minHeight: 100,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        cursor: "pointer",
        fontSize: 14,
      }}
    >
      {message}
    </div>
  );
};

const Wrapped = withStreamlitConnection(ClipboardImage);
ReactDOM.render(<Wrapped />, document.getElementById("root")); 