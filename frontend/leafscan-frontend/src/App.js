import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    setImage(e.target.files[0]);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append("image", image);

    try {
      const response = await axios.post(
        "https://<your-backend-url>/predict",
        formData
      );
      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert("Error uploading image");
    }
  };

  return (
    <div className="App">
      <h1>🌿 LeafScan</h1>
      <input type="file" accept="image/*" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload & Analyze</button>
      {result && (
        <div>
          <h3>Disease: {result.disease}</h3>
          <p>Solution: {result.solution}</p>
        </div>
      )}
    </div>
  );
}

export default App;
