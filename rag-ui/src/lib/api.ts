import type { AskRequest, AskResponse, HealthResponse } from '../types';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000').replace(/\/$/, '');

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }

  return (await response.json()) as T;
}

export function getApiBaseUrl(): string {
  return API_BASE_URL;
}

export function checkHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>('/health');
}

export function askQuestion(payload: AskRequest): Promise<AskResponse> {
  return requestJson<AskResponse>('/ask', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}
