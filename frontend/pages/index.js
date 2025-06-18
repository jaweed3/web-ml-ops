import { use, useEffect, useState } from "react";
import { fetchAndSetAll, fetchJson } from "./api/fetch-helper";

export default function Home() {
  const initialValues = {
    pregnancies: "",
    glucose: "",
    bloodPressure: "",
    skinThickness: "",
    insulin: "",
    bmi: "",
    diabetesPedigree: "",
    age: ""
  };

  const [values, setValues] = useState(initialValues);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {

    const {name, value} = e.target;
    setValues({
      ...values,
      [name]: value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault()

    setLoading(true); // Mulai Loading
    setError(null); // reset error sebelumnya

    try {
      // endpoint API
      const API_URL = "";

      const responseData = await fetchJson(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(values)
      });

      console.log('Data Berhasil Dikirim!', responseData);
      alert('Data Berhasil Dikirim! Respon : ' + JSON.stringify(responseData));
      setValues(initialValues);
    } catch (err) {
      console.error('Terjadi kesalahan saat mengirim data: ', err);
      setError('Gagal mengirim data : ' + err.message);
    } finally {
      setLoading(false); // Selesai loading
    }
  };

  return (
    <main style={{ padding: 40 }}>
      <h1>Prediksi Umur</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Pregnancies: </label>
          <input
          type="number"
          name="pregnancies"
          value={values.pregnancies}
          onChange={handleInputChange}
          placeholder="Pregnancies Rate"
          required
          />
        </div>

        <div>
          <label>Glucose Rate</label>
          <input
          type="number"
          name="glucose"
          value={values.glucose}
          onChange={handleInputChange}
          placeholder="Glucose Rate"
          required
          />
        </div>

        <div>
          <label>Blood Pressure</label>
          <input
          type="number"
          name="bloodPressure"
          value={values.bloodPressure}
          onChange={handleInputChange}
          placeholder="Blood Pressure"
          required
          />
        </div>

        <div>
          <label>Skin Thickness</label>
          <input
          type="number"
          name="skinThickness"
          value={values.skinThickness}
          onChange={handleInputChange}
          placeholder="Skin Thickness"
          required
          />
        </div>

        <div>
          <label>Insulin</label>
          <input
          type="number"
          name="insulin"
          value={values.insulin}
          onChange={handleInputChange}
          placeholder="Insulin Rate"
          required
          />
        </div>

        <div>
          <label>BMI Rate</label>
          <input
          type="number"
          step={0.01}
          name="bmi"
          value={values.bmi}
          onChange={handleInputChange}
          placeholder="BMI Rate"
          required
          />
        </div>

        <div>
          <label>Diabetes Pedigree</label>
          <input
          type="number"
          name="diabetesPedigree"
          value={values.diabetesPedigree}
          onChange={handleInputChange}
          placeholder="Diabetes Pedigree"
          required
          />
        </div>

        <div>
          <label>Age</label>
          <input
          type="number"
          name="age"
          value={values.age}
          onChange={handleInputChange}
          placeholder="Your Age"
          required
          />
        </div>

        <button type="submit" disabled={loading}>
          {loading ? 'Mengirim...' : 'Kirim Data'}
        </button>

        {error && <p style={{ color: 'red' }}>{error}</p>}
      </form>
    </main>
  );
}