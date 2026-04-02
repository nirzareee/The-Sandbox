import { useEffect } from 'react';
import { UI } from '../state/actions';

export default function Toast({ toasts, dispatch }) {
  return (
    <div className="toast-container">
      {toasts.map(toast => (
        <ToastItem key={toast.id} toast={toast} dispatch={dispatch} />
      ))}
    </div>
  );
}

function ToastItem({ toast, dispatch }) {
  useEffect(() => {
    const timer = setTimeout(() => {
      dispatch({ type: UI.REMOVE_TOAST, data: toast.id });
    }, 3000);
    return () => clearTimeout(timer);
  }, [toast.id, dispatch]);

  return <div className="toast">{toast.message}</div>;
}
