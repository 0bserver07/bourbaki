import { useState, useCallback, useRef, useEffect } from 'react';
import type { HistoryItem, WorkingState } from '../components/index.js';
import type { AgentConfig, AgentEvent, DoneEvent } from '../agent/index.js';

const BACKEND_URL = process.env.BOURBAKI_BACKEND_URL || 'http://localhost:8000';

// ============================================================================
// Types
// ============================================================================

export interface RunQueryResult {
  answer: string;
}

export interface UseAgentRunnerResult {
  // State
  history: HistoryItem[];
  workingState: WorkingState;
  error: string | null;
  isProcessing: boolean;
  sessionId: string | null;

  // Actions
  runQuery: (query: string) => Promise<RunQueryResult | undefined>;
  cancelExecution: () => void;
  setError: (error: string | null) => void;
  setSessionId: (id: string | null) => void;
  loadRestoredHistory: (messages: Array<{ role: string; content: string }>) => void;
  clearHistory: () => void;
}

// ============================================================================
// Hook
// ============================================================================

export function useAgentRunner(
  agentConfig: AgentConfig
): UseAgentRunnerResult {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [workingState, setWorkingState] = useState<WorkingState>({ status: 'idle' });
  const workingStateRef = useRef<WorkingState>({ status: 'idle' });
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const sessionIdRef = useRef<string | null>(null);

  // Keep ref in sync so runQuery always reads the latest value
  useEffect(() => { sessionIdRef.current = sessionId; }, [sessionId]);

  const abortControllerRef = useRef<AbortController | null>(null);

  // Only trigger a re-render when the working state actually changes
  const updateWorkingState = useCallback((next: WorkingState) => {
    const prev = workingStateRef.current;
    if (prev.status === next.status && (
      next.status !== 'tool' || (prev.status === 'tool' && prev.toolName === next.toolName)
    ) && (
      next.status !== 'answering' || (prev.status === 'answering' && prev.startTime === next.startTime)
    )) {
      return; // No change
    }
    workingStateRef.current = next;
    setWorkingState(next);
  }, []);

  // Build model string for the backend. The model may already contain
  // a provider prefix (e.g. "openrouter:deepseek/deepseek-r1-0528:free")
  // so we must avoid double-prefixing.
  const buildModelStr = useCallback((): string => {
    const m = agentConfig.model ?? 'gpt-4o';
    if (m.includes(':')) return m;  // Already prefixed
    if (agentConfig.modelProvider) return `${agentConfig.modelProvider}:${m}`;
    return `openai:${m}`;
  }, [agentConfig.model, agentConfig.modelProvider]);

  // Lazily ensure a backend session exists. Called before each query
  // and on mount — retries automatically if the backend was down earlier.
  const ensureSession = useCallback(async (): Promise<string | null> => {
    if (sessionIdRef.current) return sessionIdRef.current;

    const modelStr = buildModelStr();

    try {
      const res = await fetch(`${BACKEND_URL}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelStr }),
      });
      const data = await res.json();
      if (data.id) {
        sessionIdRef.current = data.id;
        setSessionId(data.id);
        return data.id;
      }
    } catch {
      // Backend not available
    }
    return null;
  }, [buildModelStr]);

  // Attempt session creation on mount (best-effort)
  useEffect(() => { ensureSession(); }, [ensureSession]);

  // Helper to update the last (processing) history item
  const updateLastHistoryItem = useCallback((
    updater: (item: HistoryItem) => Partial<HistoryItem>
  ) => {
    setHistory(prev => {
      const last = prev[prev.length - 1];
      if (!last || last.status !== 'processing') return prev;
      return [...prev.slice(0, -1), { ...last, ...updater(last) }];
    });
  }, []);

  // Handle agent events
  const handleEvent = useCallback((event: AgentEvent) => {
    switch (event.type) {
      case 'thinking':
        updateWorkingState({ status: 'thinking' });
        updateLastHistoryItem(item => ({
          events: [...item.events, {
            id: `thinking-${Date.now()}`,
            event,
            completed: true,
          }],
        }));
        break;

      case 'tool_start': {
        const toolId = `tool-${event.tool}-${Date.now()}`;
        updateWorkingState({ status: 'tool', toolName: event.tool });
        updateLastHistoryItem(item => ({
          activeToolId: toolId,
          events: [...item.events, {
            id: toolId,
            event,
            completed: false,
          }],
        }));
        break;
      }

      case 'tool_end':
        updateWorkingState({ status: 'thinking' });
        updateLastHistoryItem(item => ({
          activeToolId: undefined,
          events: item.events.map(e =>
            e.id === item.activeToolId
              ? { ...e, completed: true, endEvent: event }
              : e
          ),
        }));
        break;

      case 'tool_error':
        updateWorkingState({ status: 'thinking' });
        updateLastHistoryItem(item => ({
          activeToolId: undefined,
          events: item.events.map(e =>
            e.id === item.activeToolId
              ? { ...e, completed: true, endEvent: event }
              : e
          ),
        }));
        break;

      case 'answer_start':
        updateWorkingState({ status: 'answering', startTime: Date.now() });
        break;

      case 'done': {
        const doneEvent = event as DoneEvent;
        updateLastHistoryItem(item => ({
          answer: doneEvent.answer,
          status: 'complete' as const,
          duration: item.startTime ? Date.now() - item.startTime : undefined,
        }));
        updateWorkingState({ status: 'idle' });
        break;
      }
    }
  }, [updateLastHistoryItem, updateWorkingState]);

  // Run a query through the agent
  const runQuery = useCallback(async (query: string): Promise<RunQueryResult | undefined> => {
    // Create abort controller for this execution
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    // Track the final answer to return
    let finalAnswer: string | undefined;

    // Add to history immediately
    const itemId = Date.now().toString();
    const startTime = Date.now();
    setHistory(prev => [...prev, {
      id: itemId,
      query,
      events: [],
      answer: '',
      status: 'processing',
      startTime,
    }]);

    setError(null);
    updateWorkingState({ status: 'thinking' });

    try {
      // Ensure we have a session (retries if earlier attempt failed)
      const currentSessionId = await ensureSession();

      // Build model string in provider:model format for the Python backend
      const modelStr = buildModelStr();

      const response = await fetch(`${BACKEND_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          model: modelStr,
          session_id: currentSessionId,
        }),
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`Backend error: ${response.status} ${response.statusText}`);
      }

      // Parse SSE stream
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      const processLines = (lines: string[]) => {
        let currentEventType: string | null = null;
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEventType = line.slice(7).trim();
          } else if (line.startsWith('data: ') && currentEventType) {
            try {
              const event = JSON.parse(line.slice(6)) as AgentEvent;
              if (event.type === 'done') {
                finalAnswer = (event as DoneEvent).answer;
              }
              handleEvent(event);
            } catch {
              // Skip malformed JSON lines
            }
            currentEventType = null;
          }
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        processLines(lines);
      }

      // Process any remaining data in the buffer after stream ends
      if (buffer.trim()) {
        processLines(buffer.split('\n'));
      }

      // Return the answer if we got one
      if (finalAnswer) {
        return { answer: finalAnswer };
      }
    } catch (e) {
      // Handle abort gracefully - mark as interrupted, not error
      if (e instanceof Error && e.name === 'AbortError') {
        setHistory(prev => {
          const last = prev[prev.length - 1];
          if (!last || last.status !== 'processing') return prev;
          return [...prev.slice(0, -1), { ...last, status: 'interrupted' }];
        });
        updateWorkingState({ status: 'idle' });
        return undefined;
      }

      const errorMsg = e instanceof Error ? e.message : String(e);
      setError(errorMsg);
      // Mark the history item as error
      setHistory(prev => {
        const last = prev[prev.length - 1];
        if (!last || last.status !== 'processing') return prev;
        return [...prev.slice(0, -1), { ...last, status: 'error' }];
      });
      updateWorkingState({ status: 'idle' });
      return undefined;
    } finally {
      abortControllerRef.current = null;
    }
  }, [buildModelStr, ensureSession, handleEvent, updateWorkingState]);

  // Cancel the current execution
  const cancelExecution = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    // Mark current processing item as interrupted
    setHistory(prev => {
      const last = prev[prev.length - 1];
      if (!last || last.status !== 'processing') return prev;
      return [...prev.slice(0, -1), { ...last, status: 'interrupted' }];
    });
    updateWorkingState({ status: 'idle' });
  }, [updateWorkingState]);

  // Check if currently processing
  const isProcessing = history.length > 0 && history[history.length - 1].status === 'processing';

  // Load history from a restored session
  const loadRestoredHistory = useCallback((messages: Array<{ role: string; content: string }>) => {
    const restoredHistory: HistoryItem[] = [];
    let pendingQuery: string | null = null;

    for (const msg of messages) {
      if (msg.role === 'user') {
        if (pendingQuery !== null) {
          // Previous query had no answer
          restoredHistory.push({
            id: `restored-${restoredHistory.length}`,
            query: pendingQuery,
            events: [],
            answer: '',
            status: 'complete',
          });
        }
        pendingQuery = msg.content;
      } else if (msg.role === 'assistant' && pendingQuery !== null) {
        restoredHistory.push({
          id: `restored-${restoredHistory.length}`,
          query: pendingQuery,
          events: [],
          answer: msg.content,
          status: 'complete',
        });
        pendingQuery = null;
      }
    }

    if (pendingQuery !== null) {
      restoredHistory.push({
        id: `restored-${restoredHistory.length}`,
        query: pendingQuery,
        events: [],
        answer: '',
        status: 'complete',
      });
    }

    setHistory(restoredHistory);
  }, []);

  // Clear display history (for /new command)
  const clearHistory = useCallback(() => {
    setHistory([]);
  }, []);

  return {
    history,
    workingState,
    error,
    isProcessing,
    sessionId,
    runQuery,
    cancelExecution,
    setError,
    setSessionId,
    loadRestoredHistory,
    clearHistory,
  };
}
