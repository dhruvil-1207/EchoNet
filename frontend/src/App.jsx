import React, { useState, useRef } from 'react';
import axios from 'axios';
import { API_BASE_URL } from './config';
import './index.css'; // Importing our separate stylesheet

export default function App() {
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [status, setStatus] = useState("Ready");
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    audioChunksRef.current = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await sendAudioToModelBackend(audioBlob);
      };

      mediaRecorderRef.current.start();
      setRecording(true);
      setStatus("EchoNet is listening carefully...");
    } catch (err) {
      console.error("Microphone access initialization error:", err);
      setStatus("Error: Browser microphone permissions blocked or unsupported.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setRecording(false);
      setStatus("Processing audio arrays through model nodes...");
    }
  };

  const sendAudioToModelBackend = async (audioBlob) => {
    const formData = new FormData();
    formData.append("file", audioBlob, "user_voice.webm");

    try {
      const response = await axios.post(`${API_BASE_URL}/api/transcribe`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      if (response.data.transcript) {
        setTranscript(response.data.transcript);
        setStatus("Transcription successfully parsed!");
      } else {
        setTranscript("[No legible tokens emitted by model layers]");
        setStatus("Complete with low confidence.");
      }
    } catch (error) {
      console.error("Networking error during inference sequence:", error);
      setStatus("Error: Failed to communicate with the model backend.");
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(transcript);
    alert("Transcript text successfully mapped to clipboard!");
  };

  const downloadTranscript = () => {
    const element = document.createElement("a");
    const file = new Blob([transcript], { type: 'text/plain;charset=utf-8' });
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

        <button onClick={copyToClipboard} disabled={!transcript} className="btn btn-utility">
          Copy Text
        </button>
        
        <button onClick={downloadTranscript} disabled={!transcript} className="btn btn-utility">
          Download .txt
        </button>
      </div>

      <h3 className="transcript-header">Output Transcript Canvas</h3>
      <div className={`transcript-display ${transcript ? "text-active" : "text-placeholder"}`}>
        {transcript || "Speak clearly through your mic... your custom neural sequence transcript will print here out live."}
      </div>
    </div>
  );
}