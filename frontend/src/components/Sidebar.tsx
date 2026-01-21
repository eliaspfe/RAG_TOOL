export default function Sidebar() {
  return (
    <aside className="sidebar">
      <h2>Datenquellen</h2>

      <div className="section">
        <label>PDFs hochladen</label>
        <input type="file" accept="application/pdf" multiple />
        <button>In VektorDB speichern</button>
      </div>

      <div className="section">
        <label>Link hinzufügen</label>
        <input type="text" placeholder="https://..." />
        <button>Link speichern</button>
      </div>

      <div className="section">
        <label>Status</label>
        <div className="status">Keine laufende Verarbeitung</div>
        <button className="secondary">Index neu aufbauen</button>
      </div>
    </aside>
  );
}
