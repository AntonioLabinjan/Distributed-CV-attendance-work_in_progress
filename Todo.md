## 🛠️ TODO – Upcoming Improvements

- [x] ~~Zamjena `print()` s `logging` modulom + log fajlovi za greške i info~~
- [ ] Bolji `try/except` handling u nodu i serveru (posebno mrežni pozivi)
- [x] ~~Dodati `/ping` i `/heartbeat` rute za health monitoring servera i nodova~~
- [x] ~~Async ili threaded slanje embeddinga za nižu latenciju (BAD IDEA; EVENT BASED SENDING AKO SU EMBEDDINZI DOVOLJNO RAZLIČITI)~~
- [ ] Snapshot spremanje slike prilikom slanja embeddinga (debug/dataset)
- [ ] CLI argumenti u `node.py` (`--node_id`, `--server`, `--cam_index`, itd.)
- [ ] Failover mehanizam: čuvanje embeddinga offline ako server nedostupan
- [x] ~~Live dashboard `/nodes` za pregled statusa svih nodova~~
- [ ] Test skripte za sve rute + `pytest` test suite
- [ ] (Future) API token autentikacija za sigurnost komunikacije
- [x] Automatski refresh dataseta
- [ ] Better event based features
- [ ] Find alternative for regular Python queue
