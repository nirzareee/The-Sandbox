import { useEffect, useRef, useState } from 'react';

// Each likert question carries its own scale labels — labels are tailored
// to the question's actual semantic (enjoyability, difficulty, naturalness,
// likelihood) instead of a generic agree/disagree axis.
const LIKERT_SCALES = {
  enjoyment:    ['Not at all', 'Slightly', 'Moderately', 'Very', 'Extremely'],
  difficulty:   ['Very easy', 'Easy', 'Neutral', 'Difficult', 'Very difficult'],
  naturalness:  ['Very robotic', 'Robotic', 'Mixed', 'Natural', 'Very natural'],
  likelihood:   ['Definitely not', 'Probably not', 'Maybe', 'Probably yes', 'Definitely yes'],
};

const SURVEY_QUESTIONS = [
  { id: 'q1', text: 'How enjoyable was the game?', type: 'likert', scale: 'enjoyment', required: true },
  { id: 'q2', text: 'How difficult was it to identify the imposter?', type: 'likert', scale: 'difficulty', required: true },
  { id: 'q3', text: 'How difficult was it to distinguish humans from AI?', type: 'likert', scale: 'difficulty', required: true },
  { id: 'q4', text: "How natural did the AI players' conversation feel?", type: 'likert', scale: 'naturalness', required: true },
  { id: 'q5', text: 'Did you change your strategy after identifying potential AI players?', type: 'yesno', required: true },
  { id: 'q6', text: 'What strategies did you use to identify AI players?', type: 'checkbox', dependsOnYes: 'q5', options: [
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
  { id: 'q10', text: 'Would you play again?', type: 'likert', scale: 'likelihood', required: true },
  { id: 'q11', text: 'Any additional feedback?', type: 'textarea' },
];

function likertOptions(scaleKey) {
  const labels = LIKERT_SCALES[scaleKey] || LIKERT_SCALES.enjoyment;
  return labels.map((desc, i) => ({ value: String(i + 1), label: String(i + 1), desc }));
}

// Build a payload that always carries every question_id so research analysis
// can tell "actively skipped" apart from "didn't see this question". Empty
// answer becomes "" for free-text or yesno/likert, [] for checkbox, null when
// the question was hidden by a dependency that wasn't satisfied.
function buildPayload(responses) {
  const out = {};
  for (const q of SURVEY_QUESTIONS) {
    if (q.dependsOnYes && responses[q.dependsOnYes] !== 'yes') {
      out[q.id] = null; // not applicable
      continue;
    }
    const v = responses[q.id];
    if (q.type === 'checkbox') {
      out[q.id] = Array.isArray(v) ? v : [];
    } else {
      out[q.id] = v == null ? '' : v;
    }
  }
  return out;
}

function isAnswered(q, value) {
  if (q.type === 'checkbox') return Array.isArray(value) && value.length > 0;
  return value != null && value !== '';
}

export default function Survey({ send, timerSeconds, timerActive, surveySubmitted }) {
  const [responses, setResponses] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [showErrors, setShowErrors] = useState(false);
  // Refs for the latest values, so the auto-submit-on-unmount path uses
  // current state without re-creating the effect on every keystroke.
  const responsesRef = useRef(responses);
  const submittedRef = useRef(false);
  responsesRef.current = responses;
  submittedRef.current = submitted || surveySubmitted;

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

  function visibleQuestions() {
    return SURVEY_QUESTIONS.filter(
      q => !q.dependsOnYes || responses[q.dependsOnYes] === 'yes'
    );
  }

  function missingRequired() {
    return visibleQuestions().filter(
      q => q.required && !isAnswered(q, responses[q.id])
    );
  }

  function submit({ partial = false } = {}) {
    if (submittedRef.current) return;
    if (!partial) {
      const missing = missingRequired();
      if (missing.length > 0) {
        setShowErrors(true);
        // Scroll to the first missing question for visibility.
        const first = document.getElementById('survey-' + missing[0].id);
        if (first) first.scrollIntoView({ behavior: 'smooth', block: 'center' });
        return;
      }
    }
    const payload = buildPayload(responsesRef.current);
    if (partial) payload._partial = true;
    send({ action: 'SURVEY', content: JSON.stringify(payload) });
    setSubmitted(true);
  }

  // Best-effort partial save when the page is closing. The backend treats
  // any payload with `_partial: true` as a snapshot rather than a final
  // submission — at minimum we get whatever the user had typed.
  useEffect(() => {
    function handleUnload() {
      if (submittedRef.current) return;
      const payload = buildPayload(responsesRef.current);
      payload._partial = true;
      try {
        send({ action: 'SURVEY', content: JSON.stringify(payload) });
      } catch (_) { /* best effort */ }
    }
    window.addEventListener('beforeunload', handleUnload);
    return () => window.removeEventListener('beforeunload', handleUnload);
  }, [send]);

  // Auto-submit when the survey timer hits zero. Saves whatever's there.
  useEffect(() => {
    if (timerActive && timerSeconds === 0 && !submittedRef.current) {
      submit({ partial: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timerSeconds, timerActive]);

  const isDisabled = submitted || surveySubmitted;
  const missing = missingRequired();
  const visible = visibleQuestions();

  function renderQuestion(q) {
    const fieldErrored = showErrors && q.required && !isAnswered(q, responses[q.id]);
    switch (q.type) {
      case 'likert':
        return (
          <div className={"likert-scale" + (fieldErrored ? " survey-field-error" : "")}>
            {likertOptions(q.scale).map(opt => (
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
          <div className={"yes-no-group" + (fieldErrored ? " survey-field-error" : "")}>
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

  if (isDisabled) {
    return (
      <div className="survey-screen">
        <div className="survey-content">
          <h1>Post-Game Survey</h1>
          <div className="survey-thanks">
            <h2>Thanks — your responses were recorded.</h2>
            <p className="survey-instructions">
              Sit tight; the grand reveal is coming up next.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="survey-screen">
      <div className="survey-content">
        <h1>Post-Game Survey</h1>
        <p className="survey-instructions">
          Help our research by answering a few questions about your experience.
          Open-ended answers are <strong>especially valuable</strong> — even a sentence helps.
        </p>
        {timerActive && timerSeconds != null && (
          <p className="survey-timer">
            Time remaining: <strong>{Math.max(0, Math.floor(timerSeconds))}s</strong>
            {' '}— responses auto-save when time runs out.
          </p>
        )}

        {visible.map(q => (
          <fieldset key={q.id} id={'survey-' + q.id} className="survey-question">
            <legend>
              {q.text}
              {q.required && <span style={{ color: 'var(--danger, #ef4444)' }}> *</span>}
              {!q.required && <span style={{ color: 'var(--muted, #98a2b3)', fontWeight: 'normal' }}> (optional)</span>}
            </legend>
            {renderQuestion(q)}
          </fieldset>
        ))}

        {showErrors && missing.length > 0 && (
          <div className="survey-error-banner" role="alert">
            <strong>{missing.length} required question{missing.length === 1 ? '' : 's'} still need an answer:</strong>
            <ul style={{ margin: '6px 0 0 0', paddingLeft: '20px' }}>
              {missing.map(q => <li key={q.id}>{q.text}</li>)}
            </ul>
          </div>
        )}

        <div className="survey-footer">
          <button onClick={() => submit()}>
            Submit Survey
          </button>
        </div>
      </div>
    </div>
  );
}
