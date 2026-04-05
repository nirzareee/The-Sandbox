import { useRef, useEffect } from 'react';
import { UI } from '../state/actions.js';

/**
 * Drives the countdown timer by dispatching TICK_TIMER every second
 * while timerActive is true.
 *
 * @param {Function} dispatch - React useReducer dispatch
 * @param {boolean} timerActive - Whether the timer is currently running
 * @param {number} timerSeconds - Current seconds remaining (used only for dependency tracking)
 */
export function useTimer(dispatch, timerActive, timerSeconds) {
  const intervalRef = useRef(null);

  useEffect(() => {
    if (timerActive) {
      intervalRef.current = setInterval(() => {
        dispatch({ type: UI.TICK_TIMER });
      }, 1000);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [timerActive, dispatch]);
}

export default useTimer;
