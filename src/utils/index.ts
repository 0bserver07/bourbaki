/**
 * Bourbaki - Mathematical Reasoning Agent
 * https://github.com/0bserver07/Bourbaki
 *
 * Utility exports
 */

// Config
export { loadConfig, saveConfig, getSetting, setSetting } from './config.js';

// Environment
export {
  getApiKeyNameForProvider,
  getProviderDisplayName,
  checkApiKeyExistsForProvider,
  saveApiKeyForProvider,
} from './env.js';

// Chat History
export { LongTermChatHistory } from './long-term-chat-history.js';
export type { ConversationEntry } from './long-term-chat-history.js';

// Logging
export { logger } from './logger.js';
export type { LogEntry, LogLevel } from './logger.js';

// Input Utilities (combined from text-navigation.ts + input-key-handlers.ts)
export {
  findPrevWordStart,
  findNextWordEnd,
  getLineAndColumn,
  getCursorPosition,
  getLineStart,
  getLineEnd,
  getLineCount,
  cursorHandlers,
} from './input-utils.js';
export type { CursorContext } from './input-utils.js';

// Formatting
export { transformMarkdownTables, formatResponse } from './markdown-table.js';

// Math-specific utilities
export { translateLeanError, formatLeanError } from './lean-errors.js';
export { latexToUnicode, formatMathExpression } from './math-format.js';
