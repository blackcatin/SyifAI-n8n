import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Rate } from 'k6/metrics';

const BASE_URL = 'http://localhost:5678';
const WEBHOOK_PATH = '/webhook/predict-image';
const IMAGE_PATH = './sample_skin.webp';

const predictDuration = new Trend('predict_duration', true);
const predictErrorRate = new Rate('predict_errors');

const imageFile = open(IMAGE_PATH, 'b');

export const options = {
  vus: 1,
  duration: '1m',

  thresholds: {
    http_req_duration: ['p(95)<30000'],
    predict_errors: ['rate<0.05'],
  },
};

export default function () {

  const formData = {
    file: http.file(
      imageFile,
      'sample.jpg',
      'image/jpeg'
    ),
  };

  const res = http.post(
    `${BASE_URL}${WEBHOOK_PATH}`,
    formData,
    {
      tags: {
        name: 'PredictWebhook',
      },
      timeout: '30s',
    }
  );

  const success = check(res, {
    'status is 200': (r) => r.status === 200,
    'response has body': (r) =>
      r.body && r.body.length > 0,
  });

  predictDuration.add(res.timings.duration);
  predictErrorRate.add(!success);

  sleep(1);
}