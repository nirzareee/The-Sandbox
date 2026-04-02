import { useState } from 'react';

const SURVEY_QUESTIONS = [
  { id: 'q1', text: 'How enjoyable was the game?', type: 'likert', required: true },
  { id: 'q2', text: 'How difficult was it to identify the imposter?', type: 'likert', required: true },
  { id: 'q3', text: 'How difficult was it to distinguish humans from AI?', type: 'likert', required: true },
  { id: 'q4', text: "How natural did the AI players' conversation feel?", type: 'likert', required: true },
  { id: 'q5', text: 'Did you change your strategy after identifying potential AI players?', type: 'yesno', required: true },
  { id: 'q6', text: 'What strategies did you use to identify AI players?', type: 'checkbox', options: [
    { value: 'language_patterns', label: 'Language patterns' },
    { value: 'response_timing', label: 'Response timing' },
    { value: 'hint_quality', label: 'Hint quality' },
    { value: 'conversation_style', label: 'Conversation style' },
    { value: 'voting_behavior', label: 'Voting behavior' },
    { value: 'other', label: 'Other' },
  ]},
  { id: 'q7', text: 'Which AI player was most convincing? Why?', type: 'textarea' },
  { id: 'q8', text: 'Which AI player was least convincing? Why?', type: 'textarea' },
  { id: 'q9', text: 'What gave away that a player was AI?', type: 'textarea' },
  { id: 'q10', text: 'Would you play again?', type: 'likert', required: true },
  { id: 'q11', text: 'Any additional feedback?', type: 'textarea' },
];

const LIKERT_OPTIONS = [
  { value: '1', label: '1', desc: 'Strongly disagree' },
  { value: '2', label: '2', desc: 'Disagree' },
  { value: '3', label: '3', desc: 'Neutral' },
  { value: '4', label: '4', desc: 'Agree' },
  { value: '5', label: '5', desc: 'Strongly agree' },
];

export default function Survey({ send, timerSeconds, timerActive, surveySubmitted }) {
  const [responses, setResponses] = useState({});
  const [submitted, setSubmitted] = useState(false);

  function setResponse(id, value) {
    setResponses(prev => ({ ...prev, [id]: value }));
  }

  function toggleCheckbox(id, value) {
    setResponses(prev => {
      const current = prev[id] || [];
      if (current.includes(value)) {
        return { ...prev, [id]: current.filter(v => v !== value) };
      }
      return { ...prev, [id]: [...current, value] };
    });
  }

  function handleSubmit() {
    const missing = SURVEY_QUESTIONS.filter(q => {
      if (!q.required) return false;
      const val = responses[q.id];
      return val == null || val === '';
    });
    if (missing.length > 0) return;

    send({ action: 'SURVEY', content: JSON.stringify(responses) });
    setSubmitted(true);
  }

  const isDisabled = submitted || surveySubmitted;

  function renderQuestion(q) {
    switch (q.type) {
      case 'likert':
        return (
          <div className="likert-scale">
            {LIKERT_OPTIONS.map(opt => (
              <label key={opt.value} className="likert-option">
                <input
                  type="radio"
                  name={q.id}
                  value={opt.value}
                  checked={responses[q.id] === opt.value}
                  onChange={() => setResponse(q.id, opt.value)}
                  disabled={isDisabled}
                />
                <span className="radio-circle" />
                <span className="likert-label">{opt.label}</span>
                <span className="likert-desc">{opt.desc}</span>
              </label>
            ))}
          </div>
        );
      case 'yesno':
        return (
          <div className="yes-no-group">
            <label className="yes-no-option">
              <input
                type="radio"
                name={q.id}
                value="yes"
                checked={responses[q.id] === 'yes'}
                onChange={() => setResponse(q.id, 'yes')}
                disabled={isDisabled}
              />
              Yes
            </label>
            <label className="yes-no-option">
              <input
                type="radio"
                name={q.id}
                value="no"
                checked={responses[q.id] === 'no'}
                onChange={() => setResponse(q.id, 'no')}
                disabled={isDisabled}
              />
              No
            </label>
          </div>
        );
      case 'checkbox':
        return (
          <div className="checkbox-group">
            {q.options.map(opt => (
              <label key={opt.value} className="checkbox-label">
                <input
                  type="checkbox"
                  checked={(responses[q.id] || []).includes(opt.value)}
                  onChange={() => toggleCheckbox(q.id, opt.value)}
                  disabled={isDisabled}
                />
                <span className="checkbox-box" />
                <span>{opt.label}</span>
              </label>
            ))}
          </div>
        );
      case 'textarea':
        return (
          <textarea
            className="classification-reasoning"
            value={responses[q.id] || ''}
            onChange={e => setResponse(q.id, e.target.value)}
            placeholder="Your answer..."
            disabled={isDisabled}
          />
        );
      default:
        return null;
    }
  }

  return (
    <div className="survey-screen">
      <div className="survey-content">
        <h1>Post-Game Survey</h1>
        <p className="survey-instructions">
          Help us improve the game by answering a few questions about your experience.
        </p>

        {SURVEY_QUESTIONS.map(q => (
          <fieldset key={q.id} className="survey-question">
            <legend>
              {q.text}
              {q.required && <span style={{ color: 'var(--danger)' }}> *</span>}
            </legend>
            {renderQuestion(q)}
          </fieldset>
        ))}

        <div className="survey-footer">
          <button onClick={handleSubmit} disabled={isDisabled}>
            Submit Survey
          </button>
        </div>
      </div>
    </div>
  );
}
