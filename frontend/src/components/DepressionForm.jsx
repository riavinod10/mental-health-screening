import { useState } from "react";

const questions = [
  "Little interest or pleasure in doing things",
  "Feeling down, depressed, or hopeless",
  "Trouble falling or staying asleep, or sleeping too much",
  "Feeling tired or having little energy",
  "Poor appetite or overeating",
  "Feeling bad about yourself or that you are a failure",
  "Trouble concentrating on things",
  "Moving or speaking slowly / feeling fidgety or restless",
  "Thoughts that you would be better off dead or hurting yourself",
];

const options = [
  { label: "Not at all", value: 0 },
  { label: "Several days", value: 1 },
  { label: "More than half the days", value: 2 },
  { label: "Nearly every day", value: 3 },
];

export default function DepressionForm({ onSubmit, loading }) {
  const init = {};
  questions.forEach((_, i) => { init[`q${i+1}`] = 0; });
  const [values, setValues] = useState(init);

  function handleChange(key, val) {
    setValues(v => ({ ...v, [key]: parseInt(val) }));
  }

  return (
    <div className="form-card">
      <h2>PHQ-9 Depression Screening</h2>
      <p className="subtitle">Over the last 2 weeks, how often have you been bothered by the following?</p>

      {questions.map((q, i) => (
        <div className="question" key={i}>
          <label>{i + 1}. {q}</label>
          <div className="options">
            {options.map(o => (
              <label key={o.value} className={`option ${values[`q${i+1}`] === o.value ? 'selected' : ''}`}>
                <input
                  type="radio"
                  name={`q${i+1}`}
                  value={o.value}
                  checked={values[`q${i+1}`] === o.value}
                  onChange={e => handleChange(`q${i+1}`, e.target.value)}
                />
                {o.label}
              </label>
            ))}
          </div>
        </div>
      ))}

      <button className="submit-btn" onClick={() => onSubmit(values)} disabled={loading}>
        {loading ? "Analyzing..." : "Run Depression Screening →"}
      </button>
    </div>
  );
}