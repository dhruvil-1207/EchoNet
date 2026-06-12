import React, { useState, useRef } from 'react';
import { API_BASE_URL, API_WS_URL } from './config';
import './index.css'; // Importing our separate stylesheet

export default function App() {
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [partialTranscript, setPartialTranscript] = useState("");
  const [status, setStatus] = useState("Ready");
  
  const mediaRecorderRef = useRef(null);
  const wsRef = useRef(null);

  const startRecording = async () => {
    setTranscript("");
    setPartialTranscript("");
    setStatus("Connecting to server...");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Clean up protocol for WebSocket connection
      const wsUrl = `${API_WS_URL.replace("http://", "ws://").replace("https://", "wss://")}/api/stream-transcribe`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        
        mediaRecorderRef.current.ondataavailable = (event) => {
          if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
            ws.send(event.data);
          }
        };

        mediaRecorderRef.current.onstop = () => {
          stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorderRef.current.start(500); // Send audio slice every 500ms
        setRecording(true);
        setStatus("EchoNet is transcribing live...");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "final") {
            setTranscript(prev => prev ? prev + " " + data.text : data.text);
            setPartialTranscript("");
          } else if (data.type === "partial") {
            setPartialTranscript(data.text);
          } else if (data.type === "closed") {
            if (wsRef.current) {
              wsRef.current.close();
            }
            setStatus("Transcription completed!");
          } else if (data.type === "error") {
            setStatus(`Error: ${data.message}`);
          }
        } catch (e) {
          console.error("Failed to parse WebSocket message:", e);
        }
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        setStatus("Error: WebSocket connection failed.");
      };

      ws.onclose = () => {
        console.log("WebSocket connection closed");
      };

    } catch (err) {
      console.error("Microphone access initialization error:", err);
      setStatus("Error: Browser microphone permissions blocked or unsupported.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
      setRecording(false);
      setStatus("Finishing up live transcription...");
      
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "stop" }));
      } else {
        setStatus("Transcription completed!");
      }
    }
  };

  const copyToClipboard = () => {
    const fullText = partialTranscript 
      ? `${transcript} ${partialTranscript}`.trim() 
      : transcript;
    navigator.clipboard.writeText(fullText);
    alert("Transcript text successfully mapped to clipboard!");
  };

  const downloadTranscript = () => {
    const fullText = partialTranscript 
      ? `${transcript} ${partialTranscript}`.trim() 
      : transcript;
    const element = document.createElement("a");
    const file = new Blob([fullText], { type: 'text/plain;charset=utf-8' });
    element.href = URL.createObjectURL(file);
    element.download = "echonet_transcript.txt";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return (
    <div className="echonet-container">
      <h1 className="echonet-title">EchoNet Workspace 🎙️</h1>
      <p className="echonet-subtitle">
        System Status: <span className={recording ? "status-recording" : "status-ready"}>{status}</span>
      </p>

      <div className="controls-wrapper">
        {!recording ? (
          <button onClick={startRecording} className="btn btn-start">
            Start Recording
          </button>
        ) : (
          <button onClick={stopRecording} className="btn btn-stop">
            Stop Recording
          </button>
        )}

        <button onClick={copyToClipboard} disabled={!transcript && !partialTranscript} className="btn btn-utility">
          Copy Text
        </button>
        
        <button onClick={downloadTranscript} disabled={!transcript && !partialTranscript} className="btn btn-utility">
          Download .txt
        </button>
      </div>

      <h3 className="transcript-header">Output Transcript Canvas</h3>
      <div className={`transcript-display ${(transcript || partialTranscript) ? "text-active" : "text-placeholder"}`}>
        {transcript}
        {partialTranscript && <span className="partial-text"> {partialTranscript}</span>}
        {!transcript && !partialTranscript && "Speak clearly through your mic... your custom neural sequence transcript will print here out live."}
      </div>
    </div>
  );
}