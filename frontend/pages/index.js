import { useState } from "react";
import HomePage from "./components/homepage";
import API_BASE_URL from "@/src/api";

export const fetchJson = async (url, init = {}) => {
    const res = await fetch(url, init);
    if (!res.ok) {
        throw new Error(`${res.status}: ${await res.text()}`);
    }
    return res.json();
};

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
  const [predictionResult, setPredictionResult] = useState(null);

  // Training pipeline State
  const [trainingLoading, setTrainingLoading] = useState(null);
  const [trainingMessage, setTrainingMessage] = useState(null);
  const [trainingError, setTrainingError] = useState(null);

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
    setPredictionResult(null);

    try {
      // endpoint API
      const API_URL = "http://localhost:8080/predict";

      const responseData = await fetchJson(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(values)
      });

      console.log('Data Berhasil Dikirim!', responseData);

      if (responseData && responseData.prediction !== undefined) {
        setPredictionResult(`Prediksi dari Flask: ${responseData.prediction}`);
      } else {
        setPredictionResult('Respon dari Flask tidak memiliki prediksi yang diharapkan');
      }
      setValues(initialValues);
    } catch (err) {
      console.error('Terjadi kesalahan saat mengirim data: ', err);
      setError('Gagal mengirim data : ' + err.message);
    } finally {
      setLoading(false); // Selesai loading
    }
  };

  // handler for triggering training process.
  const handleTrainingModel = async () => {
    setTrainingLoading(true);
    setTrainingMessage(null);
    setTrainingError(null);

    try {
      const API_URL = "http://localhost:8080/train";
      const responseData = await fetchJson(API_URL, {
        method: 'GET'
      });

      console.log('Training response:', responseData);
      if (responseData && responseData.message) {
        setTrainingMessage(responseData.message);
      } else {
        setTrainingMessage("Training command sent. check flask console for detail");
      }
    } catch (err) {
      console.error('Error triggering training.', err);
      setTrainingError('Failed to trigger Training Phase.' + err.message);
    } finally {
      setTrainingLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4 font-sans">
      {/*Bagian Training Model*/}
      <div className="bg-white p-8 rounded-lg shadow-lg max-w-md w-full mb-8">
        <h1 className="test-3x1 font-bold text-center text-gray-800 mb-6">Latih Model</h1>
        <p className="text-center text-gray-600 mb-4">
          Klik tombol dibawah ini untuk memulai proses pelatihan model
        </p>
        <button
        onClick={handleTrainingModel}
        disabled={trainingLoading}
        className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out"
        >
          {trainingLoading ? 'Melatih Model ...': 'Mulai Pelatihan Model'}
        </button>
        {trainingMessage && (
          <p className="mt-4 p-3 bg-blue-100 text-blue-100 rounded-md text-center">
            {trainingMessage}
          </p>
        )}

        {trainingError && (
          <p className="mt-4 p-3 bg-red-100 text-red-800 rounded-md text-center ">
            {trainingError}
          </p>
        )}
      </div>

      {/* Bagian Prediksi Umur. */}
      <div className="bg-white p-8 rounded-lg shadow-lg max-w-md w-full">
        <h1 className="text-3x1 font-bold text-center text-gray-800 mb-6">Prediksi Diabetes</h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          {[
             {label: "Pregnancies", name: "pregnancies", type: "number", placeholder: "Pregnancies Rate"},
             {label: "Glucose", name: "glucose", type: "number", placeholder: "Glucose Rate"},
             {label: "Blood Pressure", name: "bloodPressure", type: "number", placeholder: "Blood Pressure"},
             {label: "Skin Thickness", name: "skinThickness", type: "number", placeholder: "Skin Thickness"},
             {label: "Insulin", name: "insulin", type: "number", placeholder: "Insulin"},
             {label: "BMI Rate", name: "bmi", type: "number", step:'0.01', placeholder: "BMI Rate"},
             {label: "Diabetes Pedigree", name: "diabetesPedigree",step:'0.001', type: "number", placeholder: "Diavetes Pedigree Function"},
             {label: "Age", name: "age", type: "number", placeholder: "Age"},
          ].map((field, index) => (
            <div key={index}>
              <label htmlFor={field.name} className="block txt-sm font-medium text-gray-700 mb-1">
                {field.label}:
              </label>
              <input
              id={field.name}
              type={field.type}
              name={field.name}
              value={values[field.name]}
              onChange={handleInputChange}
              placeholder={field.placeholder}
              required
              step={field.step}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              />
            </div>
          ))}

          <button
          type="submit"
          disabled={loading}
          className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none foucus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out"
          >
            {loading ? 'Mengirim...': 'Kirim Data'}
          </button>
        </form>

        {predictionResult &&(
          <p className="mt-4 p-3 bg-green-100 text-green-800 rounded-md text-center">
            {predictionResult}
          </p>
        )}

        {error && (
          <p className="mt-4 p-3 bg-red-100 text-red-800 riunded-md text-center">
            {error}
          </p>
        )}
      </div>
    </div>
  );
}