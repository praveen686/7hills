import { useAtomValue, useSetAtom } from "jotai";
import { userAtom, pageAtom, authTokenAtom } from "@/stores/auth";

/** Shared auth hook — provides user state and logout action. */
export function useAuth() {
  const user = useAtomValue(userAtom);
  const setUser = useSetAtom(userAtom);
  const setToken = useSetAtom(authTokenAtom);
  const setPage = useSetAtom(pageAtom);

  const logout = () => {
    localStorage.removeItem("ql-token");
    setUser(null);
    setToken(null);
    setPage("landing");
  };

  return { user, logout };
}
