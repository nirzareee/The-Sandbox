export default function ConnectionStatus({ connected }) {
  return (
    <div className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
      <span className="connection-dot" />
      <span className="connection-text">{connected ? 'Connected' : 'Disconnected'}</span>
    </div>
  );
}
