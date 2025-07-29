## 🛠️ TODO – Upcoming Improvements
- [ ] Auto Node Discovery (peer-to-peer pre-boot)

> Kad pokrećeš node, neka najprije provjeri postoji li konfiguracijski server (ili neki centralni announcement endpoint) s listom ostalih nodeova i servera, i sam si povuče sve potrebne parametre.
➡️ Idealno za future scaling: plug and play node deployment. Samo pokreneš node.py na novom uređaju, i bum – dio je mreže.
> idealno će bit spremit server url u env pa ga vadit iz enva...pridonosi skalabilnosti jer onda lakše dodamo novi node bez da imamo pojma di se server vrti
- [ ] Bolji `try/except` handling u nodu i serveru (posebno mrežni pozivi)
- [ ] CLI argumenti u `node.py` (`--node_id`, `--server`, `--cam_index`, itd.)
- [ ] Failover mehanizam: čuvanje embeddinga offline ako je server nedostupan
- [ ] Test skripte za sve rute + `pytest` test suite
- [ ] (Future) API token autentikacija za sigurnost komunikacije
- [ ] Dinamičko skaliranje nodeova – svaki node lokalno prati aktivnost (npr. broj lica ili kretanja) i, ako detektira neaktivnost kroz određeno vrijeme, automatski se prebacuje u *idle mode* (pauzira model i obradu); čim ponovno otkrije aktivnost, reaktivira se za punu obradu
- ✅ ~~Centralizirat FPS/Latency monitor za sve nodese (sve metrike na 1 mistu) => better; posebni log file za svaki node...nema smisla da se loga u 1 file, pa da triban scrollat ko štupido...praktičnije je ovako~~
- ✅ ~~Add latency logging on all nodes~~
- ✅ ~~Fallback mehanika.za nodes...neki lifesaver ako node crkne - > probbably neki health check unutar nodea~~
- ✅ ~~Zamjena `print()` s `logging` modulom + log fajlovi za greške i info~~
- ✅ ~~Dodati `/ping` i `/heartbeat` rute za health monitoring servera i nodova~~
- ✅ ~~Async ili threaded slanje embeddinga za nižu latenciju (BAD IDEA; EVENT BASED SENDING AKO SU EMBEDDINZI DOVOLJNO RAZLIČITI)~~
- ✅ ~~Live dashboard `/nodes` za pregled statusa svih nodova~~
- ✅ ~~Automatski refresh dataseta~~
- ✅ ~~Better event based features; napravit da ne spamma konstantno isti embedding nego nekako više graceful->implement should_classify fn~~
- ✅ ~~Find alternative for regular Python queue×(redis probbably)~~
- ✅ ~~Ne zabit da se redis mora vrtit u dockeru~~
- ✅ ~~Malo poradit na modularnosti i orkestraciji (npr. globalni imports file, centralni runner za nodese di samo prosljedin idjeve koje želin upalit i sl.)~~
- ✅ ~~Implementirat upozorenje ako je env premračan~~
- ✅ ~~Snapshot spremanje slike prilikom slanja embeddinga (debug/dataset) => NI SLUČAJNO OVO IMPLEMENTIRAT; SLIKE DETEKCIJE SE NE POHRANJUJU!!!!!~~
- ✅ ~~Implement FPS/Latency tracking~~
