Moran napravit website, bez tega nič

Dockerhub deployment: https://hub.docker.com/repository/docker/antoniolabinjan/face-rec-central_server/general
Render me ne voli :( => FUCK RENDER; sve će delat lokalno iz dockera
- todo:
- pole tega deployat image na render => TRIBAT ĆE NEKAMO DRUGAMO
- u app kodu za node ganbjat server url

Stvar dela z više kamera ... woohooo

IDEA: check if face is known (identify it implicit; do not label the class public) 
- => if it is known => grant access
- => if it is unknown => first 2 times => warning
- => third time => INTRUDER ALERT
- ali isto provjeravat dali je neko nepoznato lice bilo više puta (na više nodeova) ; npr. Ako si agregirano na k nodesa bija više od 3 puta -> that means da se neki sumnjiv mota na okolo
