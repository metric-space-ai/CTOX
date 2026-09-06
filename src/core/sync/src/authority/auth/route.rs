//! Public key discovery over an already admitted control channel. A proof binds
//! a fresh challenge to one signaling address; it never grants membership.
use super::*;
use std::collections::BTreeSet;

pub(crate) const METHOD: &str = "ctox.sync.authority.route.v1";

pub(crate) struct RouteProbe {
    request: Value,
    sender: String,
    scope: String,
    nonce: String,
    route: String,
}
impl RouteProbe {
    pub(crate) fn new(key: &SigningIdentity, scope: &str, route: &str) -> io::Result<Self> {
        let mut random = [0; 16];
        SystemRandom::new()
            .fill(&mut random)
            .map_err(|_| io::Error::other("route nonce generation failed"))?;
        let nonce = hex(&random);
        let sender = key.public_identity();
        let request = serde_json::to_value(key.sign(Body {
            version: 1,
            sender: sender.clone(),
            recipient: METHOD.into(),
            scope_id: scope.into(),
            nonce: nonce.clone(),
            kind: "route-request".into(),
            data: Value::String(route.into()),
        })?)
        .map_err(io::Error::other)?;
        Ok(Self {
            request,
            sender,
            scope: scope.into(),
            nonce,
            route: route.into(),
        })
    }
    pub(crate) fn request(&self) -> Value {
        self.request.clone()
    }
    pub(crate) fn verify(self, raw: Value, allowed: &BTreeSet<String>) -> io::Result<String> {
        let reply: Envelope = serde_json::from_value(raw).map_err(io::Error::other)?;
        verify(&reply, &self.sender, &self.scope, "route-reply")?;
        if reply.body.nonce != self.nonce
            || reply.body.data != Value::String(self.route)
            || !allowed.contains(&reply.body.sender)
        {
            return Err(invalid("unconfigured, stale or misrouted authority proof"));
        }
        Ok(reply.body.sender)
    }
}

/// A room-admitted peer may discover a public identity. This deliberately does
/// not call AuthorityNode::handle or produce an execution/membership receipt.
pub(crate) fn receive(
    key: &SigningIdentity,
    scope: &str,
    own_route: &str,
    raw: Value,
) -> io::Result<Value> {
    let request: Envelope = serde_json::from_value(raw).map_err(io::Error::other)?;
    verify(&request, METHOD, scope, "route-request")?;
    if request.body.data != Value::String(own_route.into()) {
        return Err(invalid(
            "authority route challenge addresses another signaling lifetime",
        ));
    }
    serde_json::to_value(key.sign(Body {
        version: 1,
        sender: key.public_identity(),
        recipient: request.body.sender,
        scope_id: scope.into(),
        nonce: request.body.nonce,
        kind: "route-reply".into(),
        data: request.body.data,
    })?)
    .map_err(io::Error::other)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key() -> SigningIdentity {
        SigningIdentity::from_pkcs8(&SigningIdentity::generate_pkcs8().unwrap()).unwrap()
    }
    #[test]
    fn proof_binds_configured_key_scope_route_nonce_and_recipient() {
        let caller = key();
        let voter = key();
        let stranger = key();
        let allowed = BTreeSet::from([voter.public_identity()]);
        let probe = || RouteProbe::new(&caller, "scope", "new-route").unwrap();
        let p = probe();
        let reply = receive(&voter, "scope", "new-route", p.request()).unwrap();
        assert!(probe().verify(reply.clone(), &allowed).is_err());
        assert_eq!(p.verify(reply, &allowed).unwrap(), voter.public_identity());
        let p = probe();
        assert!(receive(&voter, "other", "new-route", p.request()).is_err());
        assert!(receive(&voter, "scope", "old-route", p.request()).is_err());
        let reply = receive(&stranger, "scope", "new-route", p.request()).unwrap();
        assert!(p.verify(reply, &allowed).is_err());
        let p = probe();
        let mut reply = receive(&voter, "scope", "new-route", p.request()).unwrap();
        reply["body"]["recipient"] = stranger.public_identity().into();
        assert!(p.verify(reply, &allowed).is_err());
        let p = probe();
        let mut reply = receive(&voter, "scope", "new-route", p.request()).unwrap();
        reply["body"]["data"] = "old-route".into();
        assert!(p.verify(reply, &allowed).is_err());
    }
    #[test]
    fn route_proof_cannot_be_used_as_an_execution_reply() {
        let caller = key();
        let voter = key();
        let p = RouteProbe::new(&caller, "scope", "route").unwrap();
        let reply: Envelope =
            serde_json::from_value(receive(&voter, "scope", "route", p.request()).unwrap())
                .unwrap();
        assert!(verify(&reply, &caller.public_identity(), "scope", "reply").is_err());
    }
}
