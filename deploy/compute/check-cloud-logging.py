"""Check metadata credentials and Logging writes without printing access tokens."""
import json
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def check(open_url=urlopen):
    def metadata(path):
        request = Request('http://metadata.google.internal/computeMetadata/v1/' + path,
                          headers={'Metadata-Flavor': 'Google'})
        with open_url(request, timeout=10) as response:
            return response.read().decode()

    project = metadata('project/project-id')
    instance = metadata('instance/id')
    zone = metadata('instance/zone').rsplit('/', 1)[-1]
    token = json.loads(metadata('instance/service-accounts/default/token'))['access_token']
    payload = {
        'logName': f'projects/{project}/logs/kwonrec-deployment',
        'resource': {'type': 'gce_instance', 'labels': {
            'project_id': project, 'instance_id': instance, 'zone': zone}},
        'entries': [{'severity': 'INFO', 'textPayload': 'kwonrec deployment: Cloud Logging write verified'}],
    }
    request = Request('https://logging.googleapis.com/v2/entries:write',
                      data=json.dumps(payload).encode(), headers={
                          'Authorization': 'Bearer ' + token,
                          'Content-Type': 'application/json'})
    with open_url(request, timeout=15) as response:
        response.read()


if __name__ == '__main__':
    try:
        check()
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError):
        print('Cloud Logging preflight failed. Enable the Cloud Logging API, grant Logs Writer '
              '(roles/logging.logWriter) to the VM attached service account, and ensure the VM '
              'has logging.write or cloud-platform access scope and outbound HTTPS access. '
              'Existing containers have not been replaced.', file=sys.stderr)
        sys.exit(1)
    print('VM Cloud Logging access verified.')
