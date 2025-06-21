Moran napravit website, bez tega nič

Dockerhub deployment: https://hub.docker.com/repository/docker/antoniolabinjan/face-rec-central_server/general
Render me ne voli :( => FUCK RENDER; sve će delat lokalno iz dockera
- todo:
- pole tega deployat image na render => TRIBAT ĆE NEKAMO DRUGAMO
- u app kodu za node ganbjat server url

Stvar dela z više kamera ... woohooo
I s više osoba...ALI:
### Kamere i node konfiguracija

- Svaka kamera (node) je namijenjena skeniranju **jedne osobe istovremeno** za optimalnu točnost prepoznavanja.
- Ako se u istom kadru pojavi više lica, može doći do **nestabilnosti u detekciji i embeddingima** zbog preklapanja i trzanja frameova.
- Preporučeno je koristiti **više nodeova s fizički odvojenim kamerama**, gdje svaki pokriva zasebnu osobu ili ulaznu točku.
- nacrtat ću idealni use case

UPDT => čeka se da poštin dopelje kameru, pa onda rokamo : https://www.links.hr/hr/web-kamera-logitech-hd-webcam-c270-102500052
