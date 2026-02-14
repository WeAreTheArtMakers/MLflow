import http from "k6/http";
import { check, sleep } from "k6";

const API_URL = __ENV.API_URL || "http://host.docker.internal:8000";
const API_KEY = __ENV.API_KEY || "change-me-local-key";

export const options = {
  stages: [
    { duration: "30s", target: 10 },
    { duration: "1m", target: 40 },
    { duration: "30s", target: 0 },
  ],
  thresholds: {
    http_req_failed: ["rate<0.02"],
    http_req_duration: ["p(95)<200", "p(99)<350"],
    checks: ["rate>0.99"],
  },
};

export default function () {
  const payload = JSON.stringify({
    rows: [
      {
        hue_mean: 0.62,
        sat_mean: 0.55,
        val_mean: 0.7,
        contrast: 0.4,
        edges: 0.12,
      },
    ],
  });

  const params = {
    headers: {
      "Content-Type": "application/json",
      "x-api-key": API_KEY,
    },
  };

  const res = http.post(`${API_URL}/predict`, payload, params);
  check(res, {
    "status is 200": (r) => r.status === 200,
    "prediction has labels": (r) => {
      try {
        const body = r.json();
        return Array.isArray(body.labels) && body.labels.length > 0;
      } catch (e) {
        return false;
      }
    },
  });

  sleep(0.1);
}
