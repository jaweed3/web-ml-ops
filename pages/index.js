import { useState } from "react";

export default function Home() {
  const [age, setAge] = useState('');
  const [result, setResult] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();

    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json' },
      body: JSON.stringify({ age })
    });

    const data = await res.json();
    setResult(data.result);
  };

  return (
    <main style={{ padding: 40 }}>
      <h1>Prediksi Umur</h1>
      <form onSubmit={handleSubmit}>
        <input
        type="number"
        value={age}
        onChange={(e) => setAge(e.target.value)}
        placeholder="Masukkan Umur"
        />
        <button type="submit">Prediksi</button>
      </form>
      {result && <p>Hasil Prediksi: {result}</p>}
    </main>
  );
}