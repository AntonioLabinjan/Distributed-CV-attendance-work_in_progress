## 🛠️ TODO – Upcoming Improvements
- [ ] identificirat potencijalne bottleneckove i bugove i popravit
- [ ] napravit cleaning na nodesima malo
- [ ] Bolji `try/except` handling u nodu i serveru (posebno mrežni pozivi)
- [ ] Provat deployat server na dockerhub
- [ ] CLI argumenti u `node.py` (`--node_id`, `--server`, `--cam_index`, itd.)
- ❌ Failover mehanizam: čuvanje embeddinga offline ako je server nedostupan - > dali ovo ima smisla? Koji će mi to k realno? Server pukne i onda kad se nazad upali, klasificira osobu koja je pred po ure bila pred kameron...useless
- [ ] Test skripte za sve rute + `pytest` test suite
- [ ] Dinamičko skaliranje nodeova – svaki node lokalno prati aktivnost (npr. broj lica ili kretanja) i, ako detektira neaktivnost kroz određeno vrijeme, automatski se prebacuje u *idle mode* (pauzira model i obradu); čim ponovno otkrije aktivnost, reaktivira se za punu obradu
- [ ] (Future) API token autentikacija za sigurnost komunikacije - > dodat neki credential u node i onda kad šalje nešto na server, server brzinski provjeri dali request sadrži taj credential (vidit dali da stavljan JWT ili nešto jednostavnije) - > pošto je redis middleware, ta provjera se svakako odvija unitar classify worker jer on vadi iz redisa. Server ima listu approved tokena i brzinski provjerava dali je request poslan s validnog nodea => ovo fixat; ona varijanta s jsonima ni bila nikako dobra....jwt it is after all
Definitivno svaki node ima unique credentials...nema smisla da postoji neki common

Ovo gore je ok za početak, ali:

Taj security se svodi na 2 čitanja iz enva. Bolje da složimo da se kreira jwt pri paljenju nodea i šalje se serveru pri initial healthchecku. Server lipo ima listu allowed tokena i to brzinski provjerava svaki put kad dobije request



Bitna ideja: - >×kad to napravin, redoat docker compose da server povuče promjene

🌍 Global Time Sync TODO

Sve u UTC

Node odmah pretvori svoj lokalni timestamp u UTC.

Na serveru NIKAD ne koristi lokalno vrijeme → sve se sprema u UTC.

Server šalje heartbeat (svakih X sekundi)

Server šalje: current_utc_time.

Node izračuna koliko mu kasni ili žuri sat (drift_offset).

Node korigira svoj clock

Node NE dira sistemski sat.

Samo kod slanja eventa dodaje ili oduzima drift_offset na UTC timestamp.

Event struktura (koja se šalje serveru)

event_id (unikatni ID ili counter).

utc_timestamp (vrijeme u UTC).

local_time (vrijeme u zoni noda, samo za info/debug).

timezone_offset (koliko sati/minuta iza/ahead UTC).

drift_offset (koliko smo morali ispraviti clock).

corrected_utc (finalno vrijeme koje se koristi).

Server sve sprema po corrected_utc

Ovo je jedini timestamp koji ulazi u logiku attendancea i alertinga.

Prikaz korisniku

Kod reporta/analitike → frontend konvertira corrected_utc u lokalno vrijeme korisnika (ovisno o njihovoj zoni).

Fallback za ordering (Lamport clock)

Svaki node ima counter koji se inkrementira za svaki event.

Dodaj lamport_clock u event payload.

Ako dođe do clock problema, server složi redoslijed eventa po tom counteru.

Audit / Debug

Čuvaj sve: originalni local_time, utc_timestamp, drift_offset, corrected_utc.

To ti pomaže kad neko kaže “ali ja sam stigao na posao u 9h!!” pa možeš dokazati drift.

- ✅ ~~Napravit kompletnu pipeline schemu od početka do kraja procesa~~
- ✅ ~~Implementirat counter per node~~
- ✅ ~~Provat primjenit segmentaciju i na loading poznatih lici - > might boost precision~~  https://github.com/AntonioLabinjan/Image-Segmentator
- ✅ ~~Flask->FastAPI migracija za server; na nodesima ne triba I think~~
- ✅ ~~Roknut kod od nodesa na eng~~
- ✅[~~idealno će bit spremit server url u env pa ga vadit iz enva...pridonosi skalabilnosti jer onda lakše dodamo novi node bez da imamo pojma di se server vrti(ZAJEB...ne spreman server url nikamo jer je redis middleware između nodesa i servera) - > u env ćemo stavit: threshold distance i threshold time iz nodesa i app.run paramse iz server.py~~
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
- ✅ ~~Poigrat se malo s dockeron i njegovima mogućnostima sad kad je i novi server gore~~
-  ✅~~napravit da bbox zazeleni kad prepozna/pocrveni kad ne prepozna~~
