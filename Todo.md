## 🛠️ TODO – Upcoming Improvements

- [x] ~~Zamjena `print()` s `logging` modulom + log fajlovi za greške i info~~
- [ ] Bolji `try/except` handling u nodu i serveru (posebno mrežni pozivi)
- [x] ~~Dodati `/ping` i `/heartbeat` rute za health monitoring servera i nodova~~
- [x] ~~Async ili threaded slanje embeddinga za nižu latenciju (BAD IDEA; EVENT BASED SENDING AKO SU EMBEDDINZI DOVOLJNO RAZLIČITI)~~
- [ ] Snapshot spremanje slike prilikom slanja embeddinga (debug/dataset)
- [ ] CLI argumenti u `node.py` (`--node_id`, `--server`, `--cam_index`, itd.)
- [ ] Failover mehanizam: čuvanje embeddinga offline ako server nedostupan
- [ ] Fallback mehanika.za nodes...neki lifesaver ako node crkne
- [x] ~~Live dashboard `/nodes` za pregled statusa svih nodova~~
- [ ] Test skripte za sve rute + `pytest` test suite
- [ ] (Future) API token autentikacija za sigurnost komunikacije
- [x] ~~Automatski refresh dataseta~~
- [x] ~~Better event based features; napravit da ne spamma konstantno isti embedding nego nekako više graceful->implement should_classify fn~~
- [x] ~~Find alternative for regular Python queue×(redis probbably)~~
- [ ] Ne zabit da se redis mora vrtit u dockeru
- [x] ~~Malo poradit na modularnosti i orkestraciji (npr. globalni imports file, centralni runner za nodese di samo prosljedin idjeve koje želin upalit i sl.)~~
