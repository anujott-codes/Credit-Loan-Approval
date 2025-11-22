import axios from 'axios';

const API_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const predictCredit = async (data) => {
  try {
    const response = await api.post('/predict/credit', data);
    return response.data;
  } catch (error) {
    console.error('Error predicting credit approval:', error);
    throw error;
  }
};

export const predictLoan = async (data) => {
  try {
    const response = await api.post('/predict/loan', data);
    return response.data;
  } catch (error) {
    console.error('Error predicting loan approval:', error);
    throw error;
  }
};

export default api;
