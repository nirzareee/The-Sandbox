import { formatTime } from '../utils';

export default function TimerDisplay({ seconds, active }) {
  let cls = 'timer';
  if (active && seconds <= 10) cls += ' timer-warning';
  else if (active && seconds <= 30) cls += ' timer-low';
  return <span className={cls}>{formatTime(seconds)}</span>;
}
