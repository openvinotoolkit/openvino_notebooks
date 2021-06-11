# Try and work out the correct version of the command line tools to use
# if not explicitly specified in environment. This assumes you will be
# using the same cluster the deployment is to.

KUBERNETES_SERVER=$KUBERNETES_PORT_443_TCP_ADDR:$KUBERNETES_PORT_443_TCP_PORT

if [ x"$KUBERNETES_SERVER" != x":" ]; then
    if [ -z "$KUBECTL_VERSION" ]; then
        KUBECTL_VERSION=`(curl -s -k https://$KUBERNETES_SERVER/version | \
            python -c 'from __future__ import print_function; import sys, json; \
            info = json.loads(sys.stdin.read()); \
            info and print("%s.%s" % (info["major"], info["minor"]))') || true`
    fi
fi

if [ -z "$OC_VERSION" ]; then
    case "$KUBECTL_VERSION" in
        1.10|1.10+)
            OC_VERSION=3.10
            ;;
        1.11|1.11+)
            OC_VERSION=3.11
            ;;
        1.12|1.12+)
            OC_VERSION=4.0
            ;;
        1.13|1.13+)
            OC_VERSION=4.1
            ;;
        1.14|1.14+)
            OC_VERSION=4.2
            ;;
        1.15|1.15+)
            OC_VERSION=4.3
            ;;
        1.16|1.16+)
            OC_VERSION=4.3
            ;;
    esac
fi

export OC_VERSION
export KUBECTL_VERSION

# Setup 'oc' client configuration for the location of the OpenShift cluster.

CA_FILE="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"

if [ x"$KUBERNETES_SERVER" != x":" ]; then
    if [ -f $CA_FILE ]; then
        KUBECTL_CA_ARGS="--certificate-authority $CA_FILE"
    else
        KUBECTL_CA_ARGS="--insecure-skip-tls-verify"
    fi

    oc config set-cluster local $KUBECTL_CA_ARGS --server "https://$KUBERNETES_SERVER" 

    CONTEXT_ARGS=

    if [ x"$PROJECT_NAMESPACE" != x"" ]; then
        CONTEXT_ARGS="--namespace=$PROJECT_NAMESPACE"
    fi

    oc config set-context local --cluster local $CONTEXT_ARGS
    oc config use-context local
fi

# Now attempt to login to the OpenShift cluster. First check whether we
# inherited a user access token from shared directory volume initialised
# from an init container. If not, see if we have been passed in a user
# access token or username/password via an environment to use to login.
# Finally, see if the service account token has been mounted into the
# container.

TOKEN_DIRECTORY="/var/run/workshop"
USER_TOKEN_FILE="$TOKEN_DIRECTORY/token"
ACCT_TOKEN_FILE="/var/run/secrets/kubernetes.io/serviceaccount/token"

if [ x"$KUBERNETES_SERVER" != x":" ]; then
    if [ -f $USER_TOKEN_FILE ]; then
        oc login $KUBECTL_CA_ARGS --token `cat $USER_TOKEN_FILE` > /dev/null 2>&1
    else
        if [ x"$OPENSHIFT_TOKEN" != x"" ]; then
            oc login $KUBECTL_CA_ARGS --token "$OPENSHIFT_TOKEN" > /dev/null 2>&1
            if [ -d $TOKEN_DIRECTORY ]; then
                echo "$OPENSHIFT_TOKEN" > $USER_TOKEN_FILE.$$
                mv $USER_TOKEN_FILE.$$ $USER_TOKEN_FILE
            fi
        else
            if [ x"$OPENSHIFT_USERNAME" != x"" -a x"$OPENSHIFT_PASSWORD" != x"" ]; then
                oc login $KUBECTL_CA_ARGS -u "$OPENSHIFT_USERNAME" -p "$OPENSHIFT_PASSWORD" > /dev/null 2>&1
                if [ -d $TOKEN_DIRECTORY ]; then
                    oc whoami --show-token > $USER_TOKEN_FILE
                fi
            else
                if [ -f $ACCT_TOKEN_FILE ]; then
                    oc login $KUBECTL_CA_ARGS --token `cat $ACCT_TOKEN_FILE` > /dev/null 2>&1
                fi
            fi
        fi
    fi
fi

# If we have been supplied the name of a OpenShift project to use, change
# to that specific project, rather than rely on default selected, and try
# and create the project if it doesn't exist. We need to override the
# project namespace environment variable in this situation.

if [ x"$OPENSHIFT_PROJECT" != x"" ]; then
    oc project "$OPENSHIFT_PROJECT" || oc new-project "$OPENSHIFT_PROJECT" > /dev/null 2>&1
    export PROJECT_NAMESPACE=$OPENSHIFT_PROJECT
else
    if [ x"$PROJECT_NAMESPACE" != x"" ]; then
        oc project "$PROJECT_NAMESPACE" > /dev/null 2>&1 || true
    fi
fi
