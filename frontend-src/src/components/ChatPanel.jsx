import { useState, useEffect, useRef } from 'react';
import { getPlayerName, getAvatarColor, getInitials } from '../utils';
import { UI } from '../state/actions';

export default function ChatPanel({ players, playerId, chatMessages, dmMessages, unreadDMs, activeTab, dispatch, send, phase }) {
  const [text, setText] = useState('');
  const messagesEndRef = useRef(null);
  const inputDisabled = phase !== 'DISCUSSION' && phase !== 'REFLECTION';

  const otherPlayers = Object.keys(players).filter((pid) => pid !== playerId);

  const isGroupTab = activeTab === 'group';
  let messages = [];
  if (isGroupTab) {
    messages = chatMessages;
  } else {
    const dmPid = activeTab.slice(3);
    messages = dmMessages[dmPid] || [];
  }

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages.length]);

  function handleTabClick(tab) {
    dispatch({ type: UI.SET_ACTIVE_TAB, data: tab });
    if (tab.startsWith('dm-')) {
      const dmPid = tab.slice(3);
      dispatch({ type: UI.MARK_DM_READ, data: dmPid });
    }
  }

  function handleSend() {
    const trimmed = text.trim();
    if (!trimmed) return;

    if (isGroupTab) {
      send({ action: 'GROUP_MESSAGE', content: trimmed });
      dispatch({ type: UI.ADD_LOCAL_MESSAGE, data: { content: trimmed } });
    } else {
      const dmPid = activeTab.slice(3);
      send({ action: 'DM', target_player: dmPid, content: trimmed });
      dispatch({ type: UI.ADD_LOCAL_DM, data: { target: dmPid, content: trimmed } });
    }

    setText('');
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-tabs">
        <button
          className={`tab tab-group${activeTab === 'group' ? ' active' : ''}`}
          onClick={() => handleTabClick('group')}
        >
          <span className="tab-icon" aria-hidden>#</span>
          <span className="tab-label">Group</span>
        </button>
        {otherPlayers.map((pid) => {
          const tabId = `dm-${pid}`;
          const unread = unreadDMs[pid] || 0;
          const name = getPlayerName(players, pid);
          return (
            <button
              key={pid}
              className={`tab tab-dm${activeTab === tabId ? ' active' : ''}`}
              onClick={() => handleTabClick(tabId)}
              title={`DM ${name}`}
            >
              <span
                className="tab-avatar"
                style={{ backgroundColor: getAvatarColor(pid) }}
              >
                {getInitials(name)}
              </span>
              <span className="tab-label">{name}</span>
              {unread > 0 && <span className="unread-badge">{unread}</span>}
            </button>
          );
        })}
      </div>

      <div className="messages-area">
        {messages.map((msg, i) => {
          let cls = 'message';
          if (msg.type === 'system') cls += ' system';
          else if (msg.sender === playerId) cls += ' own';
          else cls += ' other';
          return (
            <div key={i} className={cls}>
              {msg.type !== 'system' && (
                <span className="sender">{getPlayerName(players, msg.sender)}</span>
              )}
              <span className="content">{msg.content}</span>
              {msg.timestamp && (
                <span className="timestamp">
                  {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              )}
            </div>
          );
        })}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area">
        <input
          id="chat-input"
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={inputDisabled ? 'Chat disabled' : 'Type a message...'}
          disabled={inputDisabled}
        />
        <button className="send-btn" onClick={handleSend} disabled={inputDisabled || !text.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}
