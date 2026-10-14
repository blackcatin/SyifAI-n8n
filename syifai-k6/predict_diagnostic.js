import http from 'k6/http';

const BASE_URL = 'https://widely-acapnial-aliza.ngrok-free.dev'; // domain saja, TANPA path
const WEBHOOK_PATH = '/webhook/predict-image'; // path saja, dengan slash di depan// samakan dengan yang kamu pakai
const IMAGE_PATH = './sample_skin.webp'; // samakan dengan yang kamu pakai

const imageFile = open(IMAGE_PATH, 'b');

export const options = {
  vus: 1,
  iterations: 1,
};

export default function () {
  const formData = {
    file: http.file(imageFile, 'sample.jpg', 'image/jpeg'),
  };

  const res = http.post(`${BASE_URL}${WEBHOOK_PATH}`, formData);

  console.log('STATUS CODE:', res.status);
  console.log('BODY:', res.body ? res.body.substring(0, 500) : '(kosong)');
  console.log('URL YANG DIPANGGIL:', `${BASE_URL}${WEBHOOK_PATH}`);
}